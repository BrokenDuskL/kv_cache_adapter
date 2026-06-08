#include <torch/extension.h>

#include <vector>

torch::Tensor pop_reusable_slots_cuda(
    torch::Tensor usage_count,
    torch::Tensor reusable_mask,
    torch::Tensor search_start,
    torch::Tensor blocked_slot_ids,
    int64_t count);
int64_t choose_pop_launch_warps_cuda(int64_t num_slots);

std::vector<torch::Tensor> inspect_load_requests_cuda(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor pin_count,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids);

std::vector<torch::Tensor> inspect_save_requests_cuda(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids);

void commit_load_metadata_cuda(
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
    torch::Tensor touched_usage_counts);

void commit_save_metadata_cuda(
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
    torch::Tensor final_usage_counts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pop_reusable_slots",
      &pop_reusable_slots_cuda,
      "Select reusable slots and age usage counts in-place (CUDA)",
      pybind11::arg("usage_count"),
      pybind11::arg("reusable_mask"),
      pybind11::arg("search_start"),
      pybind11::arg("blocked_slot_ids"),
      pybind11::arg("count"));
  m.def(
      "choose_pop_launch_warps",
      &choose_pop_launch_warps_cuda,
      "Choose pop_reusable_slots launch warps for a given number of slots",
      pybind11::arg("num_slots"));
  m.def(
      "inspect_load_requests",
      &inspect_load_requests_cuda,
      "Inspect load requests and prepare hit metadata (CUDA)");
  m.def(
      "inspect_save_requests",
      &inspect_save_requests_cuda,
      "Inspect save requests and prepare resident metadata (CUDA)");
  m.def(
      "commit_load_metadata",
      &commit_load_metadata_cuda,
      "Commit load metadata updates in-place (CUDA)");
  m.def(
      "commit_save_metadata",
      &commit_save_metadata_cuda,
      "Commit save metadata updates in-place (CUDA)");
}
