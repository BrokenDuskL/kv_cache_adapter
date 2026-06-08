#include <torch/extension.h>

#include <vector>

namespace {

constexpr int64_t kStateResident = 3;
constexpr int64_t kUsageCountMax = 127;

void check_tensor_1d(
    const torch::Tensor& tensor,
    torch::ScalarType scalar_type,
    const char* name) {
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has incorrect dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const char* lhs_name,
    const char* rhs_name) {
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      lhs_name,
      " and ",
      rhs_name,
      " must be on the same device");
}

torch::Tensor saturating_increment_usage(const torch::Tensor& values) {
  return torch::clamp(values.to(torch::kInt16) + 1, 0, kUsageCountMax)
      .to(torch::kUInt8);
}

}  // namespace

std::vector<torch::Tensor> inspect_load_requests_npu(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor pin_count,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(slot_state, torch::kInt64, "slot_state");
  check_tensor_1d(pin_count, torch::kInt64, "pin_count");
  check_tensor_1d(usage_count, torch::kUInt8, "usage_count");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_same_device(
      logical_to_physical,
      logical_block_ids,
      "logical_to_physical",
      "logical_block_ids");

  auto current_physical = logical_to_physical.index_select(0, logical_block_ids);
  auto resident_mask = current_physical >= 0;
  auto updated_pin_counts = torch::zeros_like(logical_block_ids);
  auto updated_usage_counts =
      torch::zeros(logical_block_ids.sizes(), usage_count.options());

  auto hit_slot_ids = current_physical.masked_select(resident_mask);
  if (hit_slot_ids.numel() == 0) {
    return {
        current_physical,
        resident_mask,
        updated_pin_counts,
        updated_usage_counts,
    };
  }

  auto hit_states = slot_state.index_select(0, hit_slot_ids);
  TORCH_CHECK(
      (hit_states == kStateResident).all().item<bool>(),
      "logical block is busy");

  auto hit_pin_counts = pin_count.index_select(0, hit_slot_ids) + 1;
  auto hit_usage_counts =
      saturating_increment_usage(usage_count.index_select(0, hit_slot_ids));
  updated_pin_counts.masked_scatter_(resident_mask, hit_pin_counts);
  updated_usage_counts.masked_scatter_(resident_mask, hit_usage_counts);
  return {
      current_physical,
      resident_mask,
      updated_pin_counts,
      updated_usage_counts,
  };
}

std::vector<torch::Tensor> inspect_save_requests_npu(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids) {
  check_tensor_1d(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_tensor_1d(slot_state, torch::kInt64, "slot_state");
  check_tensor_1d(usage_count, torch::kUInt8, "usage_count");
  check_tensor_1d(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_same_device(
      logical_to_physical,
      logical_block_ids,
      "logical_to_physical",
      "logical_block_ids");

  auto current_physical = logical_to_physical.index_select(0, logical_block_ids);
  auto existing_mask = current_physical >= 0;
  auto final_usage_counts =
      torch::ones(logical_block_ids.sizes(), usage_count.options());

  auto existing_physical = current_physical.masked_select(existing_mask);
  if (existing_physical.numel() == 0) {
    return {current_physical, existing_mask, final_usage_counts};
  }

  auto existing_states = slot_state.index_select(0, existing_physical);
  TORCH_CHECK(
      (existing_states == kStateResident).all().item<bool>(),
      "logical block is busy");

  auto existing_usage_counts =
      saturating_increment_usage(usage_count.index_select(0, existing_physical));
  final_usage_counts.masked_scatter_(existing_mask, existing_usage_counts);
  return {current_physical, existing_mask, final_usage_counts};
}

torch::Tensor pop_reusable_slots_npu(
    torch::Tensor usage_count,
    torch::Tensor reusable_mask,
    torch::Tensor search_start,
    torch::Tensor blocked_slot_ids,
    int64_t count) {
  check_tensor_1d(usage_count, torch::kUInt8, "usage_count");
  check_tensor_1d(reusable_mask, torch::kBool, "reusable_mask");
  check_tensor_1d(search_start, torch::kInt64, "search_start");
  check_tensor_1d(blocked_slot_ids, torch::kInt64, "blocked_slot_ids");
  check_same_device(usage_count, reusable_mask, "usage_count", "reusable_mask");
  check_same_device(usage_count, search_start, "usage_count", "search_start");
  check_same_device(
      usage_count, blocked_slot_ids, "usage_count", "blocked_slot_ids");
  TORCH_CHECK(search_start.numel() == 1, "search_start must contain one value");
  TORCH_CHECK(count >= 0, "count must be non-negative");

  if (count == 0) {
    return torch::empty({0}, blocked_slot_ids.options());
  }

  auto available_mask = reusable_mask.clone();
  if (blocked_slot_ids.numel() > 0) {
    available_mask.index_fill_(0, blocked_slot_ids, false);
  }

  auto available_slot_ids = torch::nonzero(available_mask).reshape(-1);
  TORCH_CHECK(
      available_slot_ids.numel() >= count,
      "No reusable actual block is available; all resident blocks are pinned");

  auto available_usage = usage_count.index_select(0, available_slot_ids)
                             .to(torch::kInt64);
  auto usage_hist = torch::bincount(available_usage);
  auto usage_prefix = torch::cumsum(usage_hist, 0);
  auto threshold = torch::nonzero(usage_prefix >= count)
                       .reshape({-1})
                       .slice(0, 0, 1)
                       .item<int64_t>();

  auto scan_order =
      (search_start[0] +
       torch::arange(
           usage_count.size(0), search_start.options().dtype(torch::kInt64))) %
      usage_count.size(0);
  auto eligible_mask =
      available_mask.index_select(0, scan_order) &
      (usage_count.index_select(0, scan_order).to(torch::kInt64) <= threshold);
  auto selected_slot_ids = scan_order.masked_select(eligible_mask).slice(0, 0, count);
  TORCH_CHECK(
      selected_slot_ids.numel() == count,
      "No reusable actual block is available; all resident blocks are pinned");

  if (threshold > 0) {
    usage_count.copy_(
        torch::clamp_min(usage_count.to(torch::kInt16) - threshold, 0)
            .to(torch::kUInt8));
  }
  auto last_selected_slot = selected_slot_ids.slice(0, count - 1, count);
  search_start.copy_(
      torch::remainder(last_selected_slot + 1, usage_count.size(0)));
  return selected_slot_ids;
}

void commit_load_metadata_npu(
    torch::Tensor logical_to_physical,
    torch::Tensor physical_to_logical,
    torch::Tensor slot_state,
    torch::Tensor pin_count,
    torch::Tensor reusable_mask,
    torch::Tensor usage_count,
    torch::Tensor evicted_logical_block_ids,
    torch::Tensor aged_slot_ids,
    torch::Tensor aged_usage_counts,
    torch::Tensor miss_logical_block_ids,
    torch::Tensor miss_physical_slot_ids,
    torch::Tensor hit_slot_ids,
    torch::Tensor hit_pin_counts,
    torch::Tensor touched_slot_ids,
    torch::Tensor touched_usage_counts) {
  (void)aged_slot_ids;
  (void)aged_usage_counts;
  if (evicted_logical_block_ids.numel() > 0) {
    logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1);
  }
  if (miss_logical_block_ids.numel() > 0) {
    logical_to_physical.index_put_(
        {miss_logical_block_ids}, miss_physical_slot_ids);
    physical_to_logical.index_put_(
        {miss_physical_slot_ids}, miss_logical_block_ids);
    slot_state.index_fill_(0, miss_physical_slot_ids, kStateResident);
    pin_count.index_fill_(0, miss_physical_slot_ids, 1);
  }
  if (hit_slot_ids.numel() > 0) {
    pin_count.index_put_({hit_slot_ids}, hit_pin_counts);
  }
  if (touched_slot_ids.numel() > 0) {
    reusable_mask.index_fill_(0, touched_slot_ids, false);
    usage_count.index_put_({touched_slot_ids}, touched_usage_counts);
  }
}

void commit_save_metadata_npu(
    torch::Tensor logical_to_physical,
    torch::Tensor physical_to_logical,
    torch::Tensor slot_state,
    torch::Tensor pin_count,
    torch::Tensor reusable_mask,
    torch::Tensor usage_count,
    torch::Tensor evicted_logical_block_ids,
    torch::Tensor aged_slot_ids,
    torch::Tensor aged_usage_counts,
    torch::Tensor logical_block_ids,
    torch::Tensor physical_slot_ids,
    torch::Tensor final_pin_counts,
    torch::Tensor final_usage_counts) {
  (void)aged_slot_ids;
  (void)aged_usage_counts;
  if (evicted_logical_block_ids.numel() > 0) {
    logical_to_physical.index_fill_(0, evicted_logical_block_ids, -1);
  }
  logical_to_physical.index_put_({logical_block_ids}, physical_slot_ids);
  physical_to_logical.index_put_({physical_slot_ids}, logical_block_ids);
  slot_state.index_fill_(0, physical_slot_ids, kStateResident);
  pin_count.index_put_({physical_slot_ids}, final_pin_counts);
  usage_count.index_put_({physical_slot_ids}, final_usage_counts);
  reusable_mask.index_put_({physical_slot_ids}, final_pin_counts == 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pop_reusable_slots",
      &pop_reusable_slots_npu,
      "Select reusable slots and age usage counts in-place (NPU)");
  m.def(
      "inspect_load_requests",
      &inspect_load_requests_npu,
      "Inspect load requests and prepare hit metadata (NPU)");
  m.def(
      "inspect_save_requests",
      &inspect_save_requests_npu,
      "Inspect save requests and prepare resident metadata (NPU)");
  m.def(
      "commit_load_metadata",
      &commit_load_metadata_npu,
      "Commit load metadata updates in-place (NPU)");
  m.def(
      "commit_save_metadata",
      &commit_save_metadata_npu,
      "Commit save metadata updates in-place (NPU)");
}
