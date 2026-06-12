#pragma once

#include <torch/extension.h>
#include <vector>

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

torch::Tensor debug_mark_blocked_slots(torch::Tensor blocked_slot_ids, int64_t num_actual_blocks);

torch::Tensor debug_count_threshold_slots(
    torch::Tensor slot_meta,
    torch::Tensor blocked_mask,
    torch::Tensor search_start,
    torch::Tensor selection_state,
    int64_t threshold);

std::vector<torch::Tensor> debug_plan_threshold_slots(
    torch::Tensor local_count_workspace,
    torch::Tensor selection_state,
    int64_t count,
    int64_t threshold);

torch::Tensor debug_collect_threshold_slots(
    torch::Tensor slot_meta,
    torch::Tensor blocked_mask,
    torch::Tensor search_start,
    torch::Tensor selection_state,
    torch::Tensor local_offset_workspace,
    torch::Tensor local_emit_workspace,
    torch::Tensor selected_slot_ids,
    int64_t threshold);

void debug_age_usage(torch::Tensor slot_meta, torch::Tensor selection_state);

void debug_finalize_selected_slots(
    torch::Tensor selection_state,
    torch::Tensor search_start,
    torch::Tensor selected_slot_ids,
    int64_t num_actual_blocks,
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
