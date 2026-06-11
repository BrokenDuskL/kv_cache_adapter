#pragma once

#include <torch/extension.h>
#include <string>
#include <vector>

std::string npu_custom_ops_build_info();

std::vector<torch::Tensor> inspect_load_requests(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids);

std::vector<torch::Tensor> inspect_save_requests(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids);

torch::Tensor pop_reusable_slots(
    torch::Tensor slot_meta,
    torch::Tensor search_start,
    torch::Tensor blocked_slot_ids,
    int64_t count);

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
    torch::Tensor miss_usage_counts);

void commit_save_metadata(
    torch::Tensor logical_to_physical,
    torch::Tensor physical_to_logical,
    torch::Tensor slot_meta,
    torch::Tensor evicted_logical_block_ids,
    torch::Tensor logical_block_ids,
    torch::Tensor physical_slot_ids,
    torch::Tensor final_pin_counts,
    torch::Tensor final_usage_counts);

void release_metadata(
    torch::Tensor logical_to_physical,
    torch::Tensor slot_meta,
    torch::Tensor logical_block_ids);
