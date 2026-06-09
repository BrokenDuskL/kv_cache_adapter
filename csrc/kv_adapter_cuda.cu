#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kUsageCountMax = 127;
constexpr int64_t kStateResident = 3;
constexpr int kWarpSize = 32;
constexpr int kMaxPopWarps = 8;
constexpr int kMetaThreads = 128;
constexpr int kMaxWarps = kMaxPopWarps;

int clamp_pop_warps(int warps) {
  return std::max(1, std::min(kMaxPopWarps, warps));
}

int read_pop_warps_override() {
  const char* value = std::getenv("KV_CACHE_ADAPTER_POP_WARPS");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  return clamp_pop_warps(std::atoi(value));
}

int choose_pop_warps(int64_t num_slots) {
  const int override_warps = read_pop_warps_override();
  if (override_warps > 0) {
    return override_warps;
  }
  if (num_slots <= 256) {
    return 1;
  }
  return 8;
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    torch::ScalarType scalar_type,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has incorrect dtype");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ bool is_blocked_slot(
    int64_t slot_id,
    const int64_t* blocked_slot_ids,
    int64_t blocked_count) {
  bool blocked = false;
  for (int64_t blocked_index = 0; blocked_index < blocked_count; ++blocked_index) {
    blocked = blocked || (blocked_slot_ids[blocked_index] == slot_id);
  }
  return blocked;
}

__global__ void pop_reusable_slots_kernel(
    uint8_t* usage_count,
    const bool* reusable_mask,
    int64_t* search_start,
    const int64_t* blocked_slot_ids,
    int64_t blocked_count,
    int64_t num_slots,
    int64_t requested_count,
    int64_t* selected_slot_ids,
    int32_t* status_out) {
  __shared__ int warp_counts[kMaxWarps];
  __shared__ int warp_bases[kMaxWarps];
  __shared__ unsigned warp_masks[kMaxWarps];
  __shared__ int threshold_shared;
  __shared__ int64_t selected_base_shared;
  __shared__ int tile_selected_total_shared;
  __shared__ int64_t selected_count_shared;
  __shared__ int64_t last_selected_slot_shared;
  __shared__ int64_t selection_limit_scan_index_shared;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  const unsigned full_mask = 0xffffffffu;
  const int64_t start = search_start[0] % num_slots;

  if (threadIdx.x == 0) {
    threshold_shared = -1;
    selected_base_shared = 0;
    tile_selected_total_shared = 0;
    selected_count_shared = 0;
    last_selected_slot_shared = start;
    selection_limit_scan_index_shared = -1;
    status_out[0] = 0;
  }
  __syncthreads();

  int64_t eligible_count = 0;
  for (int threshold = 0; threshold <= kUsageCountMax; ++threshold) {
    int local_count = 0;
    for (int64_t scan_index = threadIdx.x; scan_index < num_slots; scan_index += blockDim.x) {
      const int64_t slot_id = (start + scan_index) % num_slots;
      const bool available =
          reusable_mask[slot_id] && !is_blocked_slot(slot_id, blocked_slot_ids, blocked_count);
      local_count += available && usage_count[slot_id] == threshold ? 1 : 0;
    }

    for (int delta = kWarpSize / 2; delta > 0; delta /= 2) {
      local_count += __shfl_down_sync(full_mask, local_count, delta);
    }
    if (lane == 0) {
      warp_counts[warp] = local_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int threshold_count = 0;
      for (int warp_index = 0; warp_index < num_warps; ++warp_index) {
        threshold_count += warp_counts[warp_index];
      }
      eligible_count += threshold_count;
      if (eligible_count >= requested_count) {
        threshold_shared = threshold;
      }
    }
    __syncthreads();

    if (threshold_shared >= 0) {
      break;
    }
  }

  if (threshold_shared < 0) {
    if (threadIdx.x == 0) {
      status_out[0] = 1;
    }
    return;
  }

  for (int64_t tile_start = 0; tile_start < num_slots; tile_start += blockDim.x) {
    const int64_t scan_index = tile_start + threadIdx.x;
    const bool valid = scan_index < num_slots;
    const int64_t slot_id = valid ? (start + scan_index) % num_slots : 0;
    const bool available =
        valid && reusable_mask[slot_id] &&
        !is_blocked_slot(slot_id, blocked_slot_ids, blocked_count);
    const uint8_t current_usage = valid ? usage_count[slot_id] : static_cast<uint8_t>(0);

    if (valid && threshold_shared > 0) {
      usage_count[slot_id] = current_usage > threshold_shared
          ? static_cast<uint8_t>(current_usage - threshold_shared)
          : static_cast<uint8_t>(0);
    }

    const bool counting_tile = selection_limit_scan_index_shared < 0;
    const bool eligible = counting_tile && available && current_usage <= threshold_shared;
    const unsigned eligible_mask = __ballot_sync(full_mask, eligible);
    const int warp_count = __popc(eligible_mask);
    const unsigned lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
    const int lane_prefix = __popc(eligible_mask & lane_mask);

    if (lane == 0) {
      warp_counts[warp] = warp_count;
      warp_masks[warp] = eligible_mask;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (counting_tile) {
        int running = 0;
        for (int warp_index = 0; warp_index < num_warps; ++warp_index) {
          warp_bases[warp_index] = running;
          running += warp_counts[warp_index];
        }
        selected_base_shared = selected_count_shared;
        tile_selected_total_shared = running;

        if (selected_base_shared + running >= requested_count) {
          const int target_rank_in_tile =
              static_cast<int>(requested_count - selected_base_shared);
          int target_warp = 0;
          while (target_warp + 1 < num_warps &&
                 warp_bases[target_warp] + warp_counts[target_warp] < target_rank_in_tile) {
            ++target_warp;
          }

          const int target_rank_in_warp = target_rank_in_tile - warp_bases[target_warp];
          unsigned remaining_mask = warp_masks[target_warp];
          int target_lane = 0;
          for (int rank = 1; rank < target_rank_in_warp; ++rank) {
            remaining_mask &= (remaining_mask - 1);
          }
          target_lane = __ffs(static_cast<int>(remaining_mask)) - 1;

          selection_limit_scan_index_shared =
              tile_start + target_warp * kWarpSize + target_lane;
          last_selected_slot_shared =
              (start + selection_limit_scan_index_shared) % num_slots;
        }
      } else {
        selected_base_shared = selected_count_shared;
        tile_selected_total_shared = 0;
      }
    }
    __syncthreads();

    if (eligible) {
      const int64_t selected_index =
          selected_base_shared + warp_bases[warp] + lane_prefix;
      if (selected_index < requested_count) {
        selected_slot_ids[selected_index] = slot_id;
        if (selected_index == requested_count - 1) {
          last_selected_slot_shared = slot_id;
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (counting_tile) {
        selected_count_shared += tile_selected_total_shared;
        if (selected_count_shared > requested_count) {
          selected_count_shared = requested_count;
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    status_out[0] = selected_count_shared == requested_count ? 0 : 2;
    search_start[0] = (last_selected_slot_shared + 1) % num_slots;
  }
}

__global__ void inspect_load_requests_kernel(
    const int64_t* logical_to_physical,
    const int64_t* slot_state,
    const int64_t* pin_count,
    const uint8_t* usage_count,
    const int64_t* logical_block_ids,
    int64_t request_count,
    int64_t* current_physical_out,
    bool* resident_mask_out,
    int64_t* updated_pin_counts_out,
    uint8_t* updated_usage_counts_out,
    int32_t* status_out) {
  if (threadIdx.x == 0) {
    status_out[0] = 0;
  }
  __syncthreads();

  for (int64_t index = threadIdx.x; index < request_count; index += blockDim.x) {
    const int64_t logical_block_id = logical_block_ids[index];
    const int64_t current_physical = logical_to_physical[logical_block_id];
    current_physical_out[index] = current_physical;
    resident_mask_out[index] = false;
    updated_pin_counts_out[index] = 0;
    updated_usage_counts_out[index] = 0;

    if (current_physical < 0) {
      continue;
    }
    if (slot_state[current_physical] != kStateResident) {
      atomicExch(status_out, 1);
      continue;
    }
    resident_mask_out[index] = true;
    updated_pin_counts_out[index] = pin_count[current_physical] + 1;
    const int64_t incremented = static_cast<int64_t>(usage_count[current_physical]) + 1;
    updated_usage_counts_out[index] = incremented > kUsageCountMax
        ? static_cast<uint8_t>(kUsageCountMax)
        : static_cast<uint8_t>(incremented);
  }
}

__global__ void inspect_save_requests_kernel(
    const int64_t* logical_to_physical,
    const int64_t* slot_state,
    const uint8_t* usage_count,
    const int64_t* logical_block_ids,
    int64_t request_count,
    int64_t* current_physical_out,
    bool* existing_mask_out,
    uint8_t* final_usage_counts_out,
    int32_t* status_out) {
  if (threadIdx.x == 0) {
    status_out[0] = 0;
  }
  __syncthreads();

  for (int64_t index = threadIdx.x; index < request_count; index += blockDim.x) {
    const int64_t logical_block_id = logical_block_ids[index];
    const int64_t current_physical = logical_to_physical[logical_block_id];
    current_physical_out[index] = current_physical;
    existing_mask_out[index] = false;
    final_usage_counts_out[index] = 1;

    if (current_physical < 0) {
      continue;
    }
    if (slot_state[current_physical] != kStateResident) {
      atomicExch(status_out, 1);
      continue;
    }
    existing_mask_out[index] = true;
    const int64_t incremented = static_cast<int64_t>(usage_count[current_physical]) + 1;
    final_usage_counts_out[index] = incremented > kUsageCountMax
        ? static_cast<uint8_t>(kUsageCountMax)
        : static_cast<uint8_t>(incremented);
  }
}

__global__ void commit_load_metadata_kernel(
    int64_t* logical_to_physical,
    int64_t* physical_to_logical,
    int64_t* slot_state,
    int64_t* pin_count,
    bool* reusable_mask,
    uint8_t* usage_count,
    const int64_t* evicted_logical_block_ids,
    int64_t evicted_count,
    const int64_t* aged_slot_ids,
    const uint8_t* aged_usage_counts,
    int64_t aged_count,
    const int64_t* miss_logical_block_ids,
    const int64_t* miss_physical_slot_ids,
    int64_t miss_count,
    const int64_t* hit_slot_ids,
    const int64_t* hit_pin_counts,
    int64_t hit_count,
    const int64_t* touched_slot_ids,
    const uint8_t* touched_usage_counts,
    int64_t touched_count) {
  for (int64_t index = threadIdx.x; index < evicted_count; index += blockDim.x) {
    logical_to_physical[evicted_logical_block_ids[index]] = -1;
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < aged_count; index += blockDim.x) {
    usage_count[aged_slot_ids[index]] = aged_usage_counts[index];
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < miss_count; index += blockDim.x) {
    const int64_t logical_block_id = miss_logical_block_ids[index];
    const int64_t physical_slot_id = miss_physical_slot_ids[index];
    logical_to_physical[logical_block_id] = physical_slot_id;
    physical_to_logical[physical_slot_id] = logical_block_id;
    slot_state[physical_slot_id] = kStateResident;
    pin_count[physical_slot_id] = 1;
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < hit_count; index += blockDim.x) {
    pin_count[hit_slot_ids[index]] = hit_pin_counts[index];
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < touched_count; index += blockDim.x) {
    const int64_t slot_id = touched_slot_ids[index];
    reusable_mask[slot_id] = false;
    usage_count[slot_id] = touched_usage_counts[index];
  }
}

__global__ void commit_save_metadata_kernel(
    int64_t* logical_to_physical,
    int64_t* physical_to_logical,
    int64_t* slot_state,
    int64_t* pin_count,
    bool* reusable_mask,
    uint8_t* usage_count,
    const int64_t* evicted_logical_block_ids,
    int64_t evicted_count,
    const int64_t* aged_slot_ids,
    const uint8_t* aged_usage_counts,
    int64_t aged_count,
    const int64_t* logical_block_ids,
    const int64_t* physical_slot_ids,
    const int64_t* final_pin_counts,
    const uint8_t* final_usage_counts,
    int64_t count) {
  for (int64_t index = threadIdx.x; index < evicted_count; index += blockDim.x) {
    logical_to_physical[evicted_logical_block_ids[index]] = -1;
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < aged_count; index += blockDim.x) {
    usage_count[aged_slot_ids[index]] = aged_usage_counts[index];
  }
  __syncthreads();
  for (int64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const int64_t logical_block_id = logical_block_ids[index];
    const int64_t physical_slot_id = physical_slot_ids[index];
    const int64_t final_pin = final_pin_counts[index];
    logical_to_physical[logical_block_id] = physical_slot_id;
    physical_to_logical[physical_slot_id] = logical_block_id;
    slot_state[physical_slot_id] = kStateResident;
    pin_count[physical_slot_id] = final_pin;
    usage_count[physical_slot_id] = final_usage_counts[index];
    reusable_mask[physical_slot_id] = final_pin == 0;
  }
}

}  // namespace

torch::Tensor pop_reusable_slots_cuda(
    torch::Tensor usage_count,
    torch::Tensor reusable_mask,
    torch::Tensor search_start,
    torch::Tensor blocked_slot_ids,
    int64_t count) {
  check_cuda_tensor(usage_count, torch::kUInt8, "usage_count");
  check_cuda_tensor(reusable_mask, torch::kBool, "reusable_mask");
  check_cuda_tensor(search_start, torch::kInt64, "search_start");
  check_cuda_tensor(blocked_slot_ids, torch::kInt64, "blocked_slot_ids");
  TORCH_CHECK(
      usage_count.numel() == reusable_mask.numel(),
      "usage_count and reusable_mask must have the same length");
  TORCH_CHECK(search_start.numel() == 1, "search_start must have shape (1,)");
  TORCH_CHECK(count > 0, "count must be positive");

  auto selected_slot_ids = torch::empty(
      {count},
      torch::TensorOptions().device(usage_count.device()).dtype(torch::kInt64));
  auto available_mask = reusable_mask.clone();
  if (blocked_slot_ids.numel() > 0) {
    available_mask.index_fill_(0, blocked_slot_ids, false);
  }
  auto available_slot_ids = torch::nonzero(available_mask).reshape(-1);
  TORCH_CHECK(
      available_slot_ids.size(0) >= count,
      "No reusable actual block is available; all resident blocks are pinned");
  auto status = torch::zeros(
      {1},
      torch::TensorOptions().device(usage_count.device()).dtype(torch::kInt32));
  const int pop_threads = choose_pop_warps(usage_count.numel()) * kWarpSize;

  pop_reusable_slots_kernel<<<1, pop_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      usage_count.data_ptr<uint8_t>(),
      reusable_mask.data_ptr<bool>(),
      search_start.data_ptr<int64_t>(),
      blocked_slot_ids.data_ptr<int64_t>(),
      blocked_slot_ids.numel(),
      usage_count.numel(),
      count,
      selected_slot_ids.data_ptr<int64_t>(),
      status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return selected_slot_ids;
}

int64_t choose_pop_launch_warps_cuda(int64_t num_slots) {
  TORCH_CHECK(num_slots > 0, "num_slots must be positive");
  return choose_pop_warps(num_slots);
}

std::vector<torch::Tensor> inspect_load_requests_cuda(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor pin_count,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids) {
  check_cuda_tensor(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_cuda_tensor(slot_state, torch::kInt64, "slot_state");
  check_cuda_tensor(pin_count, torch::kInt64, "pin_count");
  check_cuda_tensor(usage_count, torch::kUInt8, "usage_count");
  check_cuda_tensor(logical_block_ids, torch::kInt64, "logical_block_ids");

  auto current_physical = torch::empty_like(logical_block_ids);
  auto resident_mask = torch::empty(
      logical_block_ids.sizes(),
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kBool));
  auto updated_pin_counts = torch::zeros_like(logical_block_ids);
  auto updated_usage_counts = torch::zeros(
      logical_block_ids.sizes(),
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kUInt8));
  auto status = torch::zeros(
      {1},
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kInt32));

  inspect_load_requests_kernel<<<1, kMetaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logical_to_physical.data_ptr<int64_t>(),
      slot_state.data_ptr<int64_t>(),
      pin_count.data_ptr<int64_t>(),
      usage_count.data_ptr<uint8_t>(),
      logical_block_ids.data_ptr<int64_t>(),
      logical_block_ids.numel(),
      current_physical.data_ptr<int64_t>(),
      resident_mask.data_ptr<bool>(),
      updated_pin_counts.data_ptr<int64_t>(),
      updated_usage_counts.data_ptr<uint8_t>(),
      status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {current_physical, resident_mask, updated_pin_counts, updated_usage_counts};
}

std::vector<torch::Tensor> inspect_save_requests_cuda(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_state,
    torch::Tensor usage_count,
    torch::Tensor logical_block_ids) {
  check_cuda_tensor(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_cuda_tensor(slot_state, torch::kInt64, "slot_state");
  check_cuda_tensor(usage_count, torch::kUInt8, "usage_count");
  check_cuda_tensor(logical_block_ids, torch::kInt64, "logical_block_ids");

  auto current_physical = torch::empty_like(logical_block_ids);
  auto existing_mask = torch::empty(
      logical_block_ids.sizes(),
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kBool));
  auto final_usage_counts = torch::zeros(
      logical_block_ids.sizes(),
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kUInt8));
  auto status = torch::zeros(
      {1},
      torch::TensorOptions().device(logical_block_ids.device()).dtype(torch::kInt32));

  inspect_save_requests_kernel<<<1, kMetaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logical_to_physical.data_ptr<int64_t>(),
      slot_state.data_ptr<int64_t>(),
      usage_count.data_ptr<uint8_t>(),
      logical_block_ids.data_ptr<int64_t>(),
      logical_block_ids.numel(),
      current_physical.data_ptr<int64_t>(),
      existing_mask.data_ptr<bool>(),
      final_usage_counts.data_ptr<uint8_t>(),
      status.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {current_physical, existing_mask, final_usage_counts};
}

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
    torch::Tensor touched_usage_counts) {
  check_cuda_tensor(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_cuda_tensor(physical_to_logical, torch::kInt64, "physical_to_logical");
  check_cuda_tensor(slot_state, torch::kInt64, "slot_state");
  check_cuda_tensor(pin_count, torch::kInt64, "pin_count");
  check_cuda_tensor(reusable_mask, torch::kBool, "reusable_mask");
  check_cuda_tensor(usage_count, torch::kUInt8, "usage_count");
  check_cuda_tensor(evicted_logical_block_ids, torch::kInt64, "evicted_logical_block_ids");
  check_cuda_tensor(aged_slot_ids, torch::kInt64, "aged_slot_ids");
  check_cuda_tensor(aged_usage_counts, torch::kUInt8, "aged_usage_counts");
  check_cuda_tensor(miss_logical_block_ids, torch::kInt64, "miss_logical_block_ids");
  check_cuda_tensor(miss_physical_slot_ids, torch::kInt64, "miss_physical_slot_ids");
  check_cuda_tensor(hit_slot_ids, torch::kInt64, "hit_slot_ids");
  check_cuda_tensor(hit_pin_counts, torch::kInt64, "hit_pin_counts");
  check_cuda_tensor(touched_slot_ids, torch::kInt64, "touched_slot_ids");
  check_cuda_tensor(touched_usage_counts, torch::kUInt8, "touched_usage_counts");

  commit_load_metadata_kernel<<<1, kMetaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logical_to_physical.data_ptr<int64_t>(),
      physical_to_logical.data_ptr<int64_t>(),
      slot_state.data_ptr<int64_t>(),
      pin_count.data_ptr<int64_t>(),
      reusable_mask.data_ptr<bool>(),
      usage_count.data_ptr<uint8_t>(),
      evicted_logical_block_ids.data_ptr<int64_t>(),
      evicted_logical_block_ids.numel(),
      aged_slot_ids.data_ptr<int64_t>(),
      aged_usage_counts.data_ptr<uint8_t>(),
      aged_slot_ids.numel(),
      miss_logical_block_ids.data_ptr<int64_t>(),
      miss_physical_slot_ids.data_ptr<int64_t>(),
      miss_logical_block_ids.numel(),
      hit_slot_ids.data_ptr<int64_t>(),
      hit_pin_counts.data_ptr<int64_t>(),
      hit_slot_ids.numel(),
      touched_slot_ids.data_ptr<int64_t>(),
      touched_usage_counts.data_ptr<uint8_t>(),
      touched_slot_ids.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    torch::Tensor final_usage_counts) {
  check_cuda_tensor(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_cuda_tensor(physical_to_logical, torch::kInt64, "physical_to_logical");
  check_cuda_tensor(slot_state, torch::kInt64, "slot_state");
  check_cuda_tensor(pin_count, torch::kInt64, "pin_count");
  check_cuda_tensor(reusable_mask, torch::kBool, "reusable_mask");
  check_cuda_tensor(usage_count, torch::kUInt8, "usage_count");
  check_cuda_tensor(evicted_logical_block_ids, torch::kInt64, "evicted_logical_block_ids");
  check_cuda_tensor(aged_slot_ids, torch::kInt64, "aged_slot_ids");
  check_cuda_tensor(aged_usage_counts, torch::kUInt8, "aged_usage_counts");
  check_cuda_tensor(logical_block_ids, torch::kInt64, "logical_block_ids");
  check_cuda_tensor(physical_slot_ids, torch::kInt64, "physical_slot_ids");
  check_cuda_tensor(final_pin_counts, torch::kInt64, "final_pin_counts");
  check_cuda_tensor(final_usage_counts, torch::kUInt8, "final_usage_counts");

  commit_save_metadata_kernel<<<1, kMetaThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      logical_to_physical.data_ptr<int64_t>(),
      physical_to_logical.data_ptr<int64_t>(),
      slot_state.data_ptr<int64_t>(),
      pin_count.data_ptr<int64_t>(),
      reusable_mask.data_ptr<bool>(),
      usage_count.data_ptr<uint8_t>(),
      evicted_logical_block_ids.data_ptr<int64_t>(),
      evicted_logical_block_ids.numel(),
      aged_slot_ids.data_ptr<int64_t>(),
      aged_usage_counts.data_ptr<uint8_t>(),
      aged_slot_ids.numel(),
      logical_block_ids.data_ptr<int64_t>(),
      physical_slot_ids.data_ptr<int64_t>(),
      final_pin_counts.data_ptr<int64_t>(),
      final_usage_counts.data_ptr<uint8_t>(),
      logical_block_ids.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void release_metadata_cuda(
    torch::Tensor logical_to_physical,
    torch::Tensor pin_count,
    torch::Tensor reusable_mask,
    torch::Tensor logical_block_ids) {
  check_cuda_tensor(logical_to_physical, torch::kInt64, "logical_to_physical");
  check_cuda_tensor(pin_count, torch::kInt64, "pin_count");
  check_cuda_tensor(reusable_mask, torch::kBool, "reusable_mask");
  check_cuda_tensor(logical_block_ids, torch::kInt64, "logical_block_ids");

  auto physical_slot_ids = logical_to_physical.index_select(0, logical_block_ids);
  auto updated_pin_counts = pin_count.index_select(0, physical_slot_ids) - 1;
  pin_count.index_put_({physical_slot_ids}, updated_pin_counts);
  auto zero_pin_slots = physical_slot_ids.masked_select(updated_pin_counts == 0);
  if (zero_pin_slots.numel() > 0) {
    reusable_mask.index_fill_(0, zero_pin_slots, true);
  }
}
