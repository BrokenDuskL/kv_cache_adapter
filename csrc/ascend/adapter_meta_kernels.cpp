#include "adapter_meta_kernels.h"

#include "kernels/adapter_metadata_kernels.h"

#include <acl/acl.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace {

bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool pop_sync_enabled() {
  static const bool enabled = env_flag_enabled("KVCA_NPU_POP_SYNC");
  return enabled;
}

void sync_stream_checked(aclrtStream stream, const char *stage) {
  const aclError status = aclrtSynchronizeStream(stream);
  TORCH_CHECK(status == ACL_SUCCESS, stage, " stream sync failed with status ", static_cast<int>(status));
}

torch::ScalarType slot_meta_scalar_type() {
#if KVCA_SLOT_META_BITS == 8
  return torch::kUInt8;
#else
  return torch::kUInt16;
#endif
}

void check_tensor_1d(
    const torch::Tensor &tensor,
    torch::ScalarType scalar_type,
    const char *name) {
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has incorrect dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor &lhs,
    const torch::Tensor &rhs,
    const char *lhs_name,
    const char *rhs_name) {
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      lhs_name,
      " and ",
      rhs_name,
      " must be on the same device");
}

int64_t block_dim_override() {
  const char *value = std::getenv("KVCA_NPU_BLOCK_DIM");
  if (value != nullptr) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && parsed > 0) {
      return static_cast<int64_t>(parsed);
    }
  }
  return 0;
}

uint32_t cap_block_dim(int64_t block_dim, int64_t count) {
  const int64_t capped = count > 0 ? std::min<int64_t>(block_dim, count) : block_dim;
  return static_cast<uint32_t>(std::max<int64_t>(1, capped));
}

uint32_t block_dim_for(int64_t count) {
  const int64_t override = block_dim_override();
  return cap_block_dim(override > 0 ? override : 1, count);
}

uint32_t pop_block_dim_for(int64_t num_actual_blocks, int64_t count) {
  const int64_t override = block_dim_override();
  if (override > 0) {
    return cap_block_dim(override, std::min<int64_t>(num_actual_blocks, count));
  }

  // Tuned from benchmark_npu_block_dim.py with count=32. Dense cases are launch-bound,
  // while tail scans start to benefit from more AICores on large slot tables.
  int64_t block_dim = 1;
  if (num_actual_blocks >= 12288) {
    block_dim = 16;
  } else if (num_actual_blocks >= 8192) {
    block_dim = 8;
  } else if (num_actual_blocks >= 3072) {
    block_dim = 2;
  }
  return cap_block_dim(block_dim, std::min<int64_t>(num_actual_blocks, count));
}

}  // namespace

std::vector<torch::Tensor> inspect_load_requests(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_same_device(logical_to_physical, slot_meta, "logical_to_physical", "slot_meta");
  check_same_device(logical_to_physical, logical_block_ids, "logical_to_physical", "logical_block_ids");

  auto current_physical = torch::empty_like(logical_block_ids);
  auto resident_mask = torch::zeros(logical_block_ids.sizes(), logical_block_ids.options().dtype(torch::kUInt8));
  auto updated_pin_counts = torch::zeros_like(logical_block_ids);
  auto updated_usage_counts = torch::zeros(logical_block_ids.sizes(), slot_meta.options());

  const c10::OptionalDeviceGuard device_guard(logical_block_ids.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(logical_block_ids.numel());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_inspect_load_requests");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_inspect_load_requests_kernel(
        block_dim,
        stream,
        logical_to_physical.data_ptr<int64_t>(),
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        logical_block_ids.data_ptr<int64_t>(),
        current_physical.data_ptr<int64_t>(),
        resident_mask.data_ptr<uint8_t>(),
        updated_pin_counts.data_ptr<int64_t>(),
        updated_usage_counts.data_ptr<kvca_slotmeta_t>(),
        static_cast<int32_t>(logical_block_ids.numel()));
    return 0;
  });
  cmd.Run();
  return {current_physical, resident_mask, updated_pin_counts, updated_usage_counts};
}

std::vector<torch::Tensor> inspect_save_requests(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_same_device(logical_to_physical, slot_meta, "logical_to_physical", "slot_meta");
  check_same_device(logical_to_physical, logical_block_ids, "logical_to_physical", "logical_block_ids");

  auto current_physical = torch::empty_like(logical_block_ids);
  auto existing_mask = torch::zeros(logical_block_ids.sizes(), logical_block_ids.options().dtype(torch::kUInt8));
  auto final_usage_counts = torch::ones(logical_block_ids.sizes(), slot_meta.options());

  const c10::OptionalDeviceGuard device_guard(logical_block_ids.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(logical_block_ids.numel());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_inspect_save_requests");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_inspect_save_requests_kernel(
        block_dim,
        stream,
        logical_to_physical.data_ptr<int64_t>(),
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        logical_block_ids.data_ptr<int64_t>(),
        current_physical.data_ptr<int64_t>(),
        existing_mask.data_ptr<uint8_t>(),
        final_usage_counts.data_ptr<kvca_slotmeta_t>(),
        static_cast<int32_t>(logical_block_ids.numel()));
    return 0;
  });
  cmd.Run();
  return {current_physical, existing_mask, final_usage_counts};
}

torch::Tensor pop_reusable_slots(
    torch::Tensor slot_meta,
    torch::Tensor search_start,
    torch::Tensor blocked_slot_ids,
    int64_t count) {
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(search_start, torch::kInt64, "search_start");
  check_tensor_1d(blocked_slot_ids, torch::kInt64, "blocked_slot_ids");
  check_same_device(slot_meta, search_start, "slot_meta", "search_start");
  check_same_device(slot_meta, blocked_slot_ids, "slot_meta", "blocked_slot_ids");
  TORCH_CHECK(search_start.numel() == 1, "search_start must contain one value");
  TORCH_CHECK(count >= 0, "count must be non-negative");

  auto selected_slot_ids = torch::empty({count}, blocked_slot_ids.options());
  if (count == 0) {
    return selected_slot_ids;
  }

  const c10::OptionalDeviceGuard device_guard(slot_meta.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  auto block_dim = pop_block_dim_for(slot_meta.numel(), count);
  if (slot_meta.numel() < 128) {
    block_dim = 1;
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_pop_reusable_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_pop_reusable_slots_kernel(
        block_dim,
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        search_start.data_ptr<int64_t>(),
        blocked_slot_ids.data_ptr<int64_t>(),
        selected_slot_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(blocked_slot_ids.numel()),
        static_cast<int32_t>(count));
    return 0;
  });
  cmd.Run();
  if (pop_sync_enabled()) {
    sync_stream_checked(stream, "pop_reusable_slots:forced_done");
  }
  return selected_slot_ids;
}

void commit_load_metadata(
    torch::Tensor logical_to_physical,
    torch::Tensor physical_to_logical,
    torch::Tensor slot_meta,
    torch::Tensor evicted_logical_block_ids,
    torch::Tensor miss_logical_block_ids,
    torch::Tensor miss_physical_slot_ids,
    torch::Tensor hit_slot_ids,
    torch::Tensor hit_pin_counts,
    torch::Tensor hit_usage_counts,
    torch::Tensor miss_usage_counts) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(physical_to_logical, torch::kInt64, "physical_to_logical");
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(evicted_logical_block_ids, torch::kInt64, "evicted_logical_block_ids");
  check_tensor_1d(miss_logical_block_ids, torch::kInt64, "miss_logical_block_ids");
  check_tensor_1d(miss_physical_slot_ids, torch::kInt64, "miss_physical_slot_ids");
  check_tensor_1d(hit_slot_ids, torch::kInt64, "hit_slot_ids");
  check_tensor_1d(hit_pin_counts, torch::kInt64, "hit_pin_counts");
  check_tensor_1d(hit_usage_counts, slot_meta_scalar_type(), "hit_usage_counts");
  check_tensor_1d(miss_usage_counts, slot_meta_scalar_type(), "miss_usage_counts");

  const c10::OptionalDeviceGuard device_guard(logical_to_physical.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(
      std::max<int64_t>(std::max(evicted_logical_block_ids.numel(), miss_logical_block_ids.numel()), hit_slot_ids.numel()));
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_commit_load_metadata");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_commit_load_metadata_kernel(
        block_dim,
        stream,
        logical_to_physical.data_ptr<int64_t>(),
        physical_to_logical.data_ptr<int64_t>(),
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        evicted_logical_block_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(evicted_logical_block_ids.numel()),
        miss_logical_block_ids.data_ptr<int64_t>(),
        miss_physical_slot_ids.data_ptr<int64_t>(),
        miss_usage_counts.data_ptr<kvca_slotmeta_t>(),
        static_cast<int32_t>(miss_logical_block_ids.numel()),
        hit_slot_ids.data_ptr<int64_t>(),
        hit_pin_counts.data_ptr<int64_t>(),
        hit_usage_counts.data_ptr<kvca_slotmeta_t>(),
        static_cast<int32_t>(hit_slot_ids.numel()));
    return 0;
  });
  cmd.Run();
}

void commit_save_metadata(
    torch::Tensor logical_to_physical,
    torch::Tensor physical_to_logical,
    torch::Tensor slot_meta,
    torch::Tensor evicted_logical_block_ids,
    torch::Tensor logical_block_ids,
    torch::Tensor physical_slot_ids,
    torch::Tensor final_pin_counts,
    torch::Tensor final_usage_counts) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(physical_to_logical, torch::kInt64, "physical_to_logical");
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(evicted_logical_block_ids, torch::kInt64, "evicted_logical_block_ids");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_tensor_1d(physical_slot_ids, torch::kInt64, "physical_slot_ids");
  check_tensor_1d(final_pin_counts, slot_meta_scalar_type(), "final_pin_counts");
  check_tensor_1d(final_usage_counts, slot_meta_scalar_type(), "final_usage_counts");

  const c10::OptionalDeviceGuard device_guard(logical_to_physical.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(std::max<int64_t>(evicted_logical_block_ids.numel(), logical_block_ids.numel()));
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_commit_save_metadata");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_commit_save_metadata_kernel(
        block_dim,
        stream,
        logical_to_physical.data_ptr<int64_t>(),
        physical_to_logical.data_ptr<int64_t>(),
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        evicted_logical_block_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(evicted_logical_block_ids.numel()),
        logical_block_ids.data_ptr<int64_t>(),
        physical_slot_ids.data_ptr<int64_t>(),
        final_pin_counts.data_ptr<kvca_slotmeta_t>(),
        final_usage_counts.data_ptr<kvca_slotmeta_t>(),
        static_cast<int32_t>(logical_block_ids.numel()));
    return 0;
  });
  cmd.Run();
}

void release_metadata(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_same_device(logical_to_physical, slot_meta, "logical_to_physical", "slot_meta");
  check_same_device(logical_to_physical, logical_block_ids, "logical_to_physical", "logical_block_ids");

  const c10::OptionalDeviceGuard device_guard(logical_to_physical.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(logical_block_ids.numel());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_release_metadata");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_release_metadata_kernel(
        block_dim,
        stream,
        logical_to_physical.data_ptr<int64_t>(),
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        logical_block_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(logical_block_ids.numel()));
    return 0;
  });
  cmd.Run();
}
