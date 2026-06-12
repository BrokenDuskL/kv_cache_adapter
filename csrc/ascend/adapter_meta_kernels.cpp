#include "adapter_meta_kernels.h"

#include "kernels/adapter_metadata_kernels.h"

#include <acl/acl.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
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

bool debug_trace_enabled() {
  static const bool enabled = env_flag_enabled("KVCA_DEBUG_TRACE");
  return enabled;
}

bool pop_sync_enabled() {
  static const bool enabled = env_flag_enabled("KVCA_NPU_POP_SYNC");
  return enabled;
}

std::string summarize_tensor(const torch::Tensor &tensor) {
  std::ostringstream oss;
  oss << "shape=(";
  for (int64_t index = 0; index < tensor.dim(); ++index) {
    if (index > 0) {
      oss << ",";
    }
    oss << tensor.size(index);
  }
  oss << "),dtype=" << tensor.scalar_type() << ",device=" << tensor.device()
      << ",ptr=" << tensor.data_ptr();
  if (tensor.dim() == 1 && tensor.numel() <= 64) {
    try {
      const auto cpu_tensor = tensor.to(torch::kCPU);
      if (cpu_tensor.scalar_type() == torch::kInt64 || cpu_tensor.scalar_type() == torch::kUInt8 ||
          cpu_tensor.scalar_type() == torch::kUInt16) {
        const auto value_tensor = cpu_tensor.to(torch::kInt64);
        oss << ",values=[";
        for (int64_t index = 0; index < value_tensor.numel(); ++index) {
          if (index > 0) {
            oss << ",";
          }
          oss << value_tensor[index].item<int64_t>();
        }
        oss << "]";
      } else {
        oss << ",values=<unsupported dtype>";
      }
    } catch (...) {
      oss << ",values=<unavailable>";
    }
  }
  return oss.str();
}

void debug_log(const char *stage, const std::string &message = "") {
  if (!debug_trace_enabled()) {
    return;
  }
  std::cerr << "[kvca-npu-debug] " << stage;
  if (!message.empty()) {
    std::cerr << " " << message;
  }
  std::cerr << std::endl;
}

void sync_stream_checked(aclrtStream stream, const char *stage, bool log_status) {
  const aclError status = aclrtSynchronizeStream(stream);
  if (log_status && debug_trace_enabled()) {
    debug_log(stage, std::string("sync_status=") + std::to_string(static_cast<int>(status)));
  }
  TORCH_CHECK(status == ACL_SUCCESS, stage, " stream sync failed with status ", static_cast<int>(status));
}

void debug_sync_stream(aclrtStream stream, const char *stage) {
  if (!debug_trace_enabled()) {
    return;
  }
  sync_stream_checked(stream, stage, true);
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

uint32_t block_dim_for(int64_t count) {
  const char *value = std::getenv("KVCA_NPU_BLOCK_DIM");
  int64_t block_dim = 1;
  if (value != nullptr) {
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && parsed > 0) {
      block_dim = static_cast<int64_t>(parsed);
    }
  }
  const int64_t capped = count > 0 ? std::min<int64_t>(block_dim, count) : block_dim;
  return static_cast<uint32_t>(std::max<int64_t>(1, capped));
}

const char *pop_debug_fields() {
  return "fields=[stage_id,threshold,num_actual_blocks,count,block_dim,search_start,"
         "selected_count,selected_threshold,local_count0,local_offset0,local_emit0,"
         "selected0,selected1,selected_last,invalid_selected_index,invalid_selected_value,"
         "blocked_count,direct_available_count,first_available_slot,second_available_slot,"
         "slot0_meta,slot0_pin,slot0_usage,slot0_blocked,slot1_meta,slot1_pin,slot1_usage,"
         "slot1_blocked,count_workspace_sum,emit_workspace_sum,max_write_end,first_oob_write_core,"
         "selected_sample0,selected_sample1,selected_sample2,selected_sample3]";
}

void debug_probe_pop_state(
    aclrtStream stream,
    const char *stage,
    int32_t stage_id,
    int32_t threshold,
    uint32_t block_dim,
    int64_t count,
    const torch::Tensor &slot_meta,
    const torch::Tensor &search_start,
    const torch::Tensor &blocked_mask,
    const torch::Tensor &selection_state,
    const torch::Tensor &local_count_workspace,
    const torch::Tensor &local_offset_workspace,
    const torch::Tensor &local_emit_workspace,
    const torch::Tensor &selected_slot_ids) {
  if (!debug_trace_enabled()) {
    return;
  }
  auto debug_workspace = torch::empty({40}, search_start.options());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_pop_reusable_slots_debug_probe");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_debug_pop_state_kernel(
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        blocked_mask.data_ptr<uint8_t>(),
        search_start.data_ptr<int64_t>(),
        selection_state.data_ptr<int64_t>(),
        local_count_workspace.data_ptr<int64_t>(),
        local_offset_workspace.data_ptr<int64_t>(),
        local_emit_workspace.data_ptr<int64_t>(),
        selected_slot_ids.data_ptr<int64_t>(),
        debug_workspace.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(count),
        threshold,
        static_cast<int32_t>(block_dim),
        stage_id);
    return 0;
  });
  cmd.Run();
  debug_sync_stream(stream, stage);
  debug_log(stage, std::string(pop_debug_fields()) + " values=" + summarize_tensor(debug_workspace));
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
  debug_sync_stream(stream, "inspect_save_requests:done");
  debug_log(
      "inspect_save_requests:state",
      "logical_to_physical=" + summarize_tensor(logical_to_physical) +
          " slot_meta=" + summarize_tensor(slot_meta) +
          " logical_block_ids=" + summarize_tensor(logical_block_ids) +
          " current_physical=" + summarize_tensor(current_physical) +
          " existing_mask=" + summarize_tensor(existing_mask) +
          " final_usage_counts=" + summarize_tensor(final_usage_counts));
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

  const auto block_dim = block_dim_for(slot_meta.numel());
  auto blocked_mask = torch::zeros({slot_meta.numel()}, slot_meta.options().dtype(torch::kUInt8));
  auto selection_state = torch::tensor({0, -1}, search_start.options());
  auto local_count_workspace = torch::zeros({block_dim}, search_start.options());
  auto local_offset_workspace = torch::zeros({block_dim}, search_start.options());
  auto local_emit_workspace = torch::zeros({block_dim}, search_start.options());
  const c10::OptionalDeviceGuard device_guard(slot_meta.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const bool trace = debug_trace_enabled();
  debug_log(
      "pop_reusable_slots:launch",
      "slot_meta=" + summarize_tensor(slot_meta) +
          " search_start=" + summarize_tensor(search_start) +
          " blocked_slot_ids=" + summarize_tensor(blocked_slot_ids) +
          " count=" + std::to_string(count) +
          " block_dim=" + std::to_string(block_dim));

  auto probe = [&](const char *stage, int32_t stage_id, int32_t threshold) {
    debug_probe_pop_state(
        stream,
        stage,
        stage_id,
        threshold,
        block_dim,
        count,
        slot_meta,
        search_start,
        blocked_mask,
        selection_state,
        local_count_workspace,
        local_offset_workspace,
        local_emit_workspace,
        selected_slot_ids);
  };

  auto run_pop_reusable_slots = [=]() {
    kvcache_ops::adapter_mark_blocked_slots_kernel(
        block_dim,
        stream,
        blocked_slot_ids.data_ptr<int64_t>(),
        blocked_mask.data_ptr<uint8_t>(),
        static_cast<int32_t>(blocked_slot_ids.numel()));
    for (int32_t threshold = 0; threshold <= KVCA_USAGE_COUNT_MAX; ++threshold) {
      kvcache_ops::adapter_count_threshold_slots_kernel(
          block_dim,
          stream,
          slot_meta.data_ptr<kvca_slotmeta_t>(),
          blocked_mask.data_ptr<uint8_t>(),
          search_start.data_ptr<int64_t>(),
          selection_state.data_ptr<int64_t>(),
          local_count_workspace.data_ptr<int64_t>(),
          static_cast<int32_t>(slot_meta.numel()),
          static_cast<int32_t>(count),
          threshold);
      kvcache_ops::adapter_plan_threshold_slots_kernel(
          stream,
          local_count_workspace.data_ptr<int64_t>(),
          local_offset_workspace.data_ptr<int64_t>(),
          local_emit_workspace.data_ptr<int64_t>(),
          selection_state.data_ptr<int64_t>(),
          static_cast<int32_t>(block_dim),
          static_cast<int32_t>(count),
          threshold);
      kvcache_ops::adapter_collect_threshold_slots_kernel(
          block_dim,
          stream,
          slot_meta.data_ptr<kvca_slotmeta_t>(),
          blocked_mask.data_ptr<uint8_t>(),
          search_start.data_ptr<int64_t>(),
          selection_state.data_ptr<int64_t>(),
          local_offset_workspace.data_ptr<int64_t>(),
          local_emit_workspace.data_ptr<int64_t>(),
          selected_slot_ids.data_ptr<int64_t>(),
          static_cast<int32_t>(slot_meta.numel()),
          threshold);
    }
    kvcache_ops::adapter_age_usage_kernel(
        block_dim,
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        selection_state.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()));
    kvcache_ops::adapter_finalize_selected_slots_kernel(
        stream,
        selection_state.data_ptr<int64_t>(),
        search_start.data_ptr<int64_t>(),
        selected_slot_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(count));
  };

  if (trace) {
    debug_sync_stream(stream, "pop_reusable_slots:after_workspace_init");
    probe("pop_reusable_slots:workspace_init_state", 0, -1);
    {
      at_npu::native::OpCommand cmd;
      cmd.Name("kv_cache_adapter_pop_reusable_slots_mark_blocked_debug");
      cmd.SetCustomHandler([=]() -> int {
        kvcache_ops::adapter_mark_blocked_slots_kernel(
            block_dim,
            stream,
            blocked_slot_ids.data_ptr<int64_t>(),
            blocked_mask.data_ptr<uint8_t>(),
            static_cast<int32_t>(blocked_slot_ids.numel()));
        return 0;
      });
      cmd.Run();
      probe("pop_reusable_slots:after_mark_blocked", 1, -1);
    }
    for (int32_t threshold = 0; threshold <= KVCA_USAGE_COUNT_MAX; ++threshold) {
      const int32_t current_threshold = threshold;
      {
        at_npu::native::OpCommand cmd;
        cmd.Name("kv_cache_adapter_pop_reusable_slots_count_debug");
        cmd.SetCustomHandler([=]() -> int {
          kvcache_ops::adapter_count_threshold_slots_kernel(
              block_dim,
              stream,
              slot_meta.data_ptr<kvca_slotmeta_t>(),
              blocked_mask.data_ptr<uint8_t>(),
              search_start.data_ptr<int64_t>(),
              selection_state.data_ptr<int64_t>(),
              local_count_workspace.data_ptr<int64_t>(),
              static_cast<int32_t>(slot_meta.numel()),
              static_cast<int32_t>(count),
              current_threshold);
          return 0;
        });
        cmd.Run();
        probe("pop_reusable_slots:after_count_threshold", 2, current_threshold);
      }
      {
        at_npu::native::OpCommand cmd;
        cmd.Name("kv_cache_adapter_pop_reusable_slots_plan_debug");
        cmd.SetCustomHandler([=]() -> int {
          kvcache_ops::adapter_plan_threshold_slots_kernel(
              stream,
              local_count_workspace.data_ptr<int64_t>(),
              local_offset_workspace.data_ptr<int64_t>(),
              local_emit_workspace.data_ptr<int64_t>(),
              selection_state.data_ptr<int64_t>(),
              static_cast<int32_t>(block_dim),
              static_cast<int32_t>(count),
              current_threshold);
          return 0;
        });
        cmd.Run();
        probe("pop_reusable_slots:after_plan_threshold", 3, current_threshold);
      }
      {
        at_npu::native::OpCommand cmd;
        cmd.Name("kv_cache_adapter_pop_reusable_slots_collect_debug");
        cmd.SetCustomHandler([=]() -> int {
          kvcache_ops::adapter_collect_threshold_slots_kernel(
              block_dim,
              stream,
              slot_meta.data_ptr<kvca_slotmeta_t>(),
              blocked_mask.data_ptr<uint8_t>(),
              search_start.data_ptr<int64_t>(),
              selection_state.data_ptr<int64_t>(),
              local_offset_workspace.data_ptr<int64_t>(),
              local_emit_workspace.data_ptr<int64_t>(),
              selected_slot_ids.data_ptr<int64_t>(),
              static_cast<int32_t>(slot_meta.numel()),
              current_threshold);
          return 0;
        });
        cmd.Run();
        probe("pop_reusable_slots:after_collect_threshold", 4, current_threshold);
      }
      if (selection_state.to(torch::kCPU)[1].item<int64_t>() >= 0) {
        break;
      }
    }
    {
      at_npu::native::OpCommand cmd;
      cmd.Name("kv_cache_adapter_pop_reusable_slots_age_debug");
      cmd.SetCustomHandler([=]() -> int {
        kvcache_ops::adapter_age_usage_kernel(
            block_dim,
            stream,
            slot_meta.data_ptr<kvca_slotmeta_t>(),
            selection_state.data_ptr<int64_t>(),
            static_cast<int32_t>(slot_meta.numel()));
        return 0;
      });
      cmd.Run();
      probe("pop_reusable_slots:after_age_usage", 5, -1);
    }
    {
      at_npu::native::OpCommand cmd;
      cmd.Name("kv_cache_adapter_pop_reusable_slots_finalize_debug");
      cmd.SetCustomHandler([=]() -> int {
        kvcache_ops::adapter_finalize_selected_slots_kernel(
            stream,
            selection_state.data_ptr<int64_t>(),
            search_start.data_ptr<int64_t>(),
            selected_slot_ids.data_ptr<int64_t>(),
            static_cast<int32_t>(slot_meta.numel()),
            static_cast<int32_t>(count));
        return 0;
      });
      cmd.Run();
      probe("pop_reusable_slots:after_finalize", 6, -1);
    }
    const auto selection_state_cpu = selection_state.to(torch::kCPU);
    const int64_t selected_count = selection_state_cpu[0].item<int64_t>();
    const int64_t selected_threshold = selection_state_cpu[1].item<int64_t>();
    const auto selected_slot_ids_cpu = selected_slot_ids.to(torch::kCPU);
    int64_t invalid_index = -1;
    int64_t invalid_value = 0;
    for (int64_t index = 0; index < selected_slot_ids_cpu.numel(); ++index) {
      const int64_t slot_id = selected_slot_ids_cpu[index].item<int64_t>();
      if (slot_id < 0 || slot_id >= slot_meta.numel()) {
        invalid_index = index;
        invalid_value = slot_id;
        break;
      }
    }
    TORCH_CHECK(
        selected_count == count && invalid_index < 0,
        "pop_reusable_slots failed to select valid slots: selected_count=",
        selected_count,
        ", selected_threshold=",
        selected_threshold,
        ", count=",
        count,
        ", invalid_index=",
        invalid_index,
        ", invalid_value=",
        invalid_value,
        ", selected_slot_ids=",
        summarize_tensor(selected_slot_ids),
        ", slot_meta=",
        summarize_tensor(slot_meta),
        ", blocked_slot_ids=",
        summarize_tensor(blocked_slot_ids));
    debug_log(
        "pop_reusable_slots:done",
        "selection_state=" + summarize_tensor(selection_state) +
            " local_count_workspace=" + summarize_tensor(local_count_workspace) +
            " local_offset_workspace=" + summarize_tensor(local_offset_workspace) +
            " local_emit_workspace=" + summarize_tensor(local_emit_workspace) +
            " selected_slot_ids=" + summarize_tensor(selected_slot_ids));
  } else {
    at_npu::native::OpCommand cmd;
    cmd.Name("kv_cache_adapter_pop_reusable_slots");
    cmd.SetCustomHandler([=]() -> int {
      run_pop_reusable_slots();
      return 0;
    });
    cmd.Run();
    if (pop_sync_enabled()) {
      sync_stream_checked(stream, "pop_reusable_slots:forced_done", false);
    }
  }
  return selected_slot_ids;
}

torch::Tensor debug_mark_blocked_slots(torch::Tensor blocked_slot_ids, int64_t num_actual_blocks) {
  check_tensor_1d(blocked_slot_ids, torch::kInt64, "blocked_slot_ids");
  TORCH_CHECK(num_actual_blocks >= 0, "num_actual_blocks must be non-negative");

  auto blocked_mask = torch::zeros({num_actual_blocks}, blocked_slot_ids.options().dtype(torch::kUInt8));
  const c10::OptionalDeviceGuard device_guard(blocked_slot_ids.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(blocked_slot_ids.numel());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_mark_blocked_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_mark_blocked_slots_kernel(
        block_dim,
        stream,
        blocked_slot_ids.data_ptr<int64_t>(),
        blocked_mask.data_ptr<uint8_t>(),
        static_cast<int32_t>(blocked_slot_ids.numel()));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_mark_blocked_slots:done", false);
  return blocked_mask;
}

torch::Tensor debug_count_threshold_slots(
    torch::Tensor slot_meta,
    torch::Tensor blocked_mask,
    torch::Tensor search_start,
    torch::Tensor selection_state,
    int64_t threshold) {
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(blocked_mask, torch::kUInt8, "blocked_mask");
  check_tensor_1d(search_start, torch::kInt64, "search_start");
  check_tensor_1d(selection_state, torch::kInt64, "selection_state");
  check_same_device(slot_meta, blocked_mask, "slot_meta", "blocked_mask");
  check_same_device(slot_meta, search_start, "slot_meta", "search_start");
  check_same_device(slot_meta, selection_state, "slot_meta", "selection_state");
  TORCH_CHECK(search_start.numel() == 1, "search_start must contain one value");
  TORCH_CHECK(selection_state.numel() == 2, "selection_state must contain selected_count and threshold");

  const auto block_dim = block_dim_for(slot_meta.numel());
  auto local_count_workspace = torch::zeros({block_dim}, search_start.options());
  const c10::OptionalDeviceGuard device_guard(slot_meta.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_count_threshold_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_count_threshold_slots_kernel(
        block_dim,
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        blocked_mask.data_ptr<uint8_t>(),
        search_start.data_ptr<int64_t>(),
        selection_state.data_ptr<int64_t>(),
        local_count_workspace.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(threshold));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_count_threshold_slots:done", false);
  return local_count_workspace;
}

std::vector<torch::Tensor> debug_plan_threshold_slots(
    torch::Tensor local_count_workspace,
    torch::Tensor selection_state,
    int64_t count,
    int64_t threshold) {
  check_tensor_1d(local_count_workspace, torch::kInt64, "local_count_workspace");
  check_tensor_1d(selection_state, torch::kInt64, "selection_state");
  check_same_device(local_count_workspace, selection_state, "local_count_workspace", "selection_state");
  TORCH_CHECK(selection_state.numel() == 2, "selection_state must contain selected_count and threshold");
  TORCH_CHECK(count >= 0, "count must be non-negative");

  auto local_offset_workspace = torch::zeros_like(local_count_workspace);
  auto local_emit_workspace = torch::zeros_like(local_count_workspace);
  const c10::OptionalDeviceGuard device_guard(local_count_workspace.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_plan_threshold_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_plan_threshold_slots_kernel(
        stream,
        local_count_workspace.data_ptr<int64_t>(),
        local_offset_workspace.data_ptr<int64_t>(),
        local_emit_workspace.data_ptr<int64_t>(),
        selection_state.data_ptr<int64_t>(),
        static_cast<int32_t>(local_count_workspace.numel()),
        static_cast<int32_t>(count),
        static_cast<int32_t>(threshold));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_plan_threshold_slots:done", false);
  return {local_offset_workspace, local_emit_workspace, selection_state};
}

torch::Tensor debug_collect_threshold_slots(
    torch::Tensor slot_meta,
    torch::Tensor blocked_mask,
    torch::Tensor search_start,
    torch::Tensor selection_state,
    torch::Tensor local_offset_workspace,
    torch::Tensor local_emit_workspace,
    torch::Tensor selected_slot_ids,
    int64_t threshold) {
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(blocked_mask, torch::kUInt8, "blocked_mask");
  check_tensor_1d(search_start, torch::kInt64, "search_start");
  check_tensor_1d(selection_state, torch::kInt64, "selection_state");
  check_tensor_1d(local_offset_workspace, torch::kInt64, "local_offset_workspace");
  check_tensor_1d(local_emit_workspace, torch::kInt64, "local_emit_workspace");
  check_tensor_1d(selected_slot_ids, torch::kInt64, "selected_slot_ids");
  check_same_device(slot_meta, blocked_mask, "slot_meta", "blocked_mask");
  check_same_device(slot_meta, search_start, "slot_meta", "search_start");
  check_same_device(slot_meta, selection_state, "slot_meta", "selection_state");
  check_same_device(slot_meta, local_offset_workspace, "slot_meta", "local_offset_workspace");
  check_same_device(slot_meta, local_emit_workspace, "slot_meta", "local_emit_workspace");
  check_same_device(slot_meta, selected_slot_ids, "slot_meta", "selected_slot_ids");
  TORCH_CHECK(search_start.numel() == 1, "search_start must contain one value");
  TORCH_CHECK(selection_state.numel() == 2, "selection_state must contain selected_count and threshold");
  TORCH_CHECK(local_offset_workspace.numel() == local_emit_workspace.numel(), "workspace sizes must match");

  const auto block_dim = static_cast<uint32_t>(local_emit_workspace.numel());
  const c10::OptionalDeviceGuard device_guard(slot_meta.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_collect_threshold_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_collect_threshold_slots_kernel(
        block_dim,
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        blocked_mask.data_ptr<uint8_t>(),
        search_start.data_ptr<int64_t>(),
        selection_state.data_ptr<int64_t>(),
        local_offset_workspace.data_ptr<int64_t>(),
        local_emit_workspace.data_ptr<int64_t>(),
        selected_slot_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()),
        static_cast<int32_t>(threshold));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_collect_threshold_slots:done", false);
  return selected_slot_ids;
}

void debug_age_usage(torch::Tensor slot_meta, torch::Tensor selection_state) {
  check_tensor_1d(slot_meta, slot_meta_scalar_type(), "slot_meta");
  check_tensor_1d(selection_state, torch::kInt64, "selection_state");
  check_same_device(slot_meta, selection_state, "slot_meta", "selection_state");
  TORCH_CHECK(selection_state.numel() == 2, "selection_state must contain selected_count and threshold");

  const c10::OptionalDeviceGuard device_guard(slot_meta.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  const auto block_dim = block_dim_for(slot_meta.numel());
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_age_usage");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_age_usage_kernel(
        block_dim,
        stream,
        slot_meta.data_ptr<kvca_slotmeta_t>(),
        selection_state.data_ptr<int64_t>(),
        static_cast<int32_t>(slot_meta.numel()));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_age_usage:done", false);
}

void debug_finalize_selected_slots(
    torch::Tensor selection_state,
    torch::Tensor search_start,
    torch::Tensor selected_slot_ids,
    int64_t num_actual_blocks,
    int64_t count) {
  check_tensor_1d(selection_state, torch::kInt64, "selection_state");
  check_tensor_1d(search_start, torch::kInt64, "search_start");
  check_tensor_1d(selected_slot_ids, torch::kInt64, "selected_slot_ids");
  check_same_device(selection_state, search_start, "selection_state", "search_start");
  check_same_device(selection_state, selected_slot_ids, "selection_state", "selected_slot_ids");
  TORCH_CHECK(selection_state.numel() == 2, "selection_state must contain selected_count and threshold");
  TORCH_CHECK(search_start.numel() == 1, "search_start must contain one value");
  TORCH_CHECK(num_actual_blocks >= 0, "num_actual_blocks must be non-negative");
  TORCH_CHECK(count >= 0, "count must be non-negative");

  const c10::OptionalDeviceGuard device_guard(selection_state.device());
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("kv_cache_adapter_debug_finalize_selected_slots");
  cmd.SetCustomHandler([=]() -> int {
    kvcache_ops::adapter_finalize_selected_slots_kernel(
        stream,
        selection_state.data_ptr<int64_t>(),
        search_start.data_ptr<int64_t>(),
        selected_slot_ids.data_ptr<int64_t>(),
        static_cast<int32_t>(num_actual_blocks),
        static_cast<int32_t>(count));
    return 0;
  });
  cmd.Run();
  sync_stream_checked(stream, "debug_finalize_selected_slots:done", false);
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
