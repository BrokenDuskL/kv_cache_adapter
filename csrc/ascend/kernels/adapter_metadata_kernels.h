#pragma once

#include <cstdint>

#ifndef KVCA_SLOT_META_BITS
#define KVCA_SLOT_META_BITS 8
#endif

#if KVCA_SLOT_META_BITS == 8
using kvca_slotmeta_t = uint8_t;
constexpr int32_t KVCA_PIN_COUNT_BITS = 4;
constexpr int32_t KVCA_USAGE_COUNT_BITS = 4;
#elif KVCA_SLOT_META_BITS == 16
using kvca_slotmeta_t = uint16_t;
constexpr int32_t KVCA_PIN_COUNT_BITS = 8;
constexpr int32_t KVCA_USAGE_COUNT_BITS = 8;
#else
#error "KVCA_SLOT_META_BITS must be 8 or 16"
#endif

constexpr int32_t KVCA_PIN_COUNT_MASK = (1 << KVCA_PIN_COUNT_BITS) - 1;
constexpr int32_t KVCA_USAGE_COUNT_SHIFT = KVCA_PIN_COUNT_BITS;
constexpr int32_t KVCA_USAGE_COUNT_MASK = (1 << KVCA_USAGE_COUNT_BITS) - 1;
constexpr int32_t KVCA_USAGE_COUNT_MAX = KVCA_USAGE_COUNT_MASK;

namespace kvcache_ops {

void adapter_inspect_load_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    uint8_t *resident_mask_out,
    int64_t *updated_pin_counts_out,
    kvca_slotmeta_t *updated_usage_counts_out,
    int32_t num_logical_ids);

void adapter_inspect_save_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    uint8_t *existing_mask_out,
    kvca_slotmeta_t *final_usage_counts_out,
    int32_t num_logical_ids);

void adapter_pop_reusable_slots_kernel(
    uint32_t block_dim,
    void *stream,
    kvca_slotmeta_t *slot_meta,
    int64_t *search_start,
    const int64_t *blocked_slot_ids,
    uint8_t *blocked_mask,
    int64_t *selection_state,
    int64_t *local_count_workspace,
    int64_t *local_offset_workspace,
    int64_t *local_emit_workspace,
    int64_t *selected_slot_ids_out,
    int32_t num_actual_blocks,
    int32_t num_blocked_slot_ids,
    int32_t count);

void adapter_mark_blocked_slots_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *blocked_slot_ids,
    uint8_t *blocked_mask,
    int32_t num_blocked_slot_ids);

void adapter_count_threshold_slots_kernel(
    uint32_t block_dim,
    void *stream,
    const kvca_slotmeta_t *slot_meta,
    const uint8_t *blocked_mask,
    const int64_t *search_start,
    const int64_t *selection_state,
    int64_t *local_count_workspace,
    int32_t num_actual_blocks,
    int32_t threshold);

void adapter_plan_threshold_slots_kernel(
    void *stream,
    const int64_t *local_count_workspace,
    int64_t *local_offset_workspace,
    int64_t *local_emit_workspace,
    int64_t *selection_state,
    int32_t block_dim,
    int32_t count,
    int32_t threshold);

void adapter_collect_threshold_slots_kernel(
    uint32_t block_dim,
    void *stream,
    const kvca_slotmeta_t *slot_meta,
    const uint8_t *blocked_mask,
    const int64_t *search_start,
    const int64_t *selection_state,
    const int64_t *local_offset_workspace,
    const int64_t *local_emit_workspace,
    int64_t *selected_slot_ids_out,
    int32_t num_actual_blocks,
    int32_t threshold);

void adapter_age_usage_kernel(
    uint32_t block_dim,
    void *stream,
    kvca_slotmeta_t *slot_meta,
    const int64_t *selection_state,
    int32_t num_actual_blocks);

void adapter_finalize_selected_slots_kernel(
    void *stream,
    const int64_t *selection_state,
    int64_t *search_start,
    int64_t *selected_slot_ids_out,
    int32_t num_actual_blocks,
    int32_t count);

void adapter_commit_load_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    int64_t *logical_to_physical,
    int64_t *physical_to_logical,
    kvca_slotmeta_t *slot_meta,
    const int64_t *evicted_logical_block_ids,
    int32_t num_evicted,
    const int64_t *miss_logical_block_ids,
    const int64_t *miss_physical_slot_ids,
    const kvca_slotmeta_t *miss_usage_counts,
    int32_t num_misses,
    const int64_t *hit_slot_ids,
    const int64_t *hit_pin_counts,
    const kvca_slotmeta_t *hit_usage_counts,
    int32_t num_hits);

void adapter_commit_save_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    int64_t *logical_to_physical,
    int64_t *physical_to_logical,
    kvca_slotmeta_t *slot_meta,
    const int64_t *evicted_logical_block_ids,
    int32_t num_evicted,
    const int64_t *logical_block_ids,
    const int64_t *physical_slot_ids,
    const kvca_slotmeta_t *final_pin_counts,
    const kvca_slotmeta_t *final_usage_counts,
    int32_t num_slots);

void adapter_release_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int32_t num_logical_ids);

}  // namespace kvcache_ops
