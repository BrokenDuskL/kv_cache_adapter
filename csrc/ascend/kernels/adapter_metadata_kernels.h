#pragma once

#include <cstdint>

namespace kvcache_ops {

#ifndef KVCA_SLOT_META_BITS
#define KVCA_SLOT_META_BITS 8
#endif

#if KVCA_SLOT_META_BITS == 8
using slotmeta_t = uint8_t;
constexpr int32_t kPinCountBits = 4;
constexpr int32_t kUsageCountBits = 4;
#elif KVCA_SLOT_META_BITS == 16
using slotmeta_t = uint16_t;
constexpr int32_t kPinCountBits = 8;
constexpr int32_t kUsageCountBits = 8;
#else
#error "KVCA_SLOT_META_BITS must be 8 or 16"
#endif

constexpr int32_t kPinCountMask = (1 << kPinCountBits) - 1;
constexpr int32_t kUsageCountShift = kPinCountBits;
constexpr int32_t kUsageCountMask = (1 << kUsageCountBits) - 1;
constexpr int32_t kUsageCountMax = kUsageCountMask;

void adapter_inspect_load_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    bool *resident_mask_out,
    int64_t *updated_pin_counts_out,
    slotmeta_t *updated_usage_counts_out,
    int32_t num_logical_ids);

void adapter_inspect_save_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    bool *existing_mask_out,
    slotmeta_t *final_usage_counts_out,
    int32_t num_logical_ids);

void adapter_pop_reusable_slots_kernel(
    uint32_t block_dim,
    void *stream,
    slotmeta_t *slot_meta,
    int64_t *search_start,
    const int64_t *blocked_slot_ids,
    bool *blocked_mask,
    int64_t *selection_state,
    int64_t *local_count_workspace,
    int64_t *local_offset_workspace,
    int64_t *local_emit_workspace,
    int64_t *selected_slot_ids_out,
    int32_t num_actual_blocks,
    int32_t num_blocked_slot_ids,
    int32_t count);

void adapter_commit_load_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    int64_t *logical_to_physical,
    int64_t *physical_to_logical,
    slotmeta_t *slot_meta,
    const int64_t *evicted_logical_block_ids,
    int32_t num_evicted,
    const int64_t *miss_logical_block_ids,
    const int64_t *miss_physical_slot_ids,
    const slotmeta_t *miss_usage_counts,
    int32_t num_misses,
    const int64_t *hit_slot_ids,
    const int64_t *hit_pin_counts,
    const slotmeta_t *hit_usage_counts,
    int32_t num_hits);

void adapter_commit_save_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    int64_t *logical_to_physical,
    int64_t *physical_to_logical,
    slotmeta_t *slot_meta,
    const int64_t *evicted_logical_block_ids,
    int32_t num_evicted,
    const int64_t *logical_block_ids,
    const int64_t *physical_slot_ids,
    const int64_t *final_pin_counts,
    const slotmeta_t *final_usage_counts,
    int32_t num_slots);

void adapter_release_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int32_t num_logical_ids);

}  // namespace kvcache_ops
