#include "adapter_metadata_kernels.h"

#include "kernel_operator.h"

namespace {

constexpr int32_t kMetaTileElems = 256;

__aicore__ inline int32_t chunk_begin(int32_t total, int32_t core_index, int32_t core_count) {
    return (total * core_index) / core_count;
}

__aicore__ inline int32_t chunk_end(int32_t total, int32_t core_index, int32_t core_count) {
    return (total * (core_index + 1)) / core_count;
}

__aicore__ inline uint32_t ceil_32_bytes(int32_t size) {
    return size % 32 == 0 ? static_cast<uint32_t>(size) : static_cast<uint32_t>(32 * (1 + (size / 32)));
}

template <typename T>
__aicore__ inline void load_tile(
    const AscendC::LocalTensor<T> &local_tensor,
    const __gm__ T *global_ptr,
    int32_t len) {
    AscendC::GlobalTensor<T> global_tensor;
    global_tensor.SetGlobalBuffer(const_cast<__gm__ T *>(global_ptr), len);
    AscendC::DataCopy(local_tensor, global_tensor, len);
}

template <typename T>
__aicore__ inline void store_tile(
    __gm__ T *global_ptr,
    const AscendC::LocalTensor<T> &local_tensor,
    int32_t len) {
    AscendC::GlobalTensor<T> global_tensor;
    global_tensor.SetGlobalBuffer(global_ptr, len);
    AscendC::DataCopy(global_tensor, local_tensor, len);
}

__aicore__ inline int32_t unpack_pin_count(kvca_slotmeta_t meta) {
    return static_cast<int32_t>(meta) & KVCA_PIN_COUNT_MASK;
}

__aicore__ inline int32_t unpack_usage_count(kvca_slotmeta_t meta) {
    return (static_cast<int32_t>(meta) >> KVCA_USAGE_COUNT_SHIFT) & KVCA_USAGE_COUNT_MASK;
}

__aicore__ inline kvca_slotmeta_t pack_slot_meta(int32_t pin_count, int32_t usage_count) {
    const int32_t clamped_pin =
        pin_count > KVCA_PIN_COUNT_MASK ? KVCA_PIN_COUNT_MASK : (pin_count < 0 ? 0 : pin_count);
    const int32_t clamped_usage =
        usage_count > KVCA_USAGE_COUNT_MASK ? KVCA_USAGE_COUNT_MASK : (usage_count < 0 ? 0 : usage_count);
    return static_cast<kvca_slotmeta_t>(clamped_pin | (clamped_usage << KVCA_USAGE_COUNT_SHIFT));
}

__aicore__ inline kvca_slotmeta_t saturating_increment_usage(kvca_slotmeta_t meta) {
    const int32_t pin_count = unpack_pin_count(meta);
    const int32_t usage_count = unpack_usage_count(meta);
    const int32_t incremented =
        usage_count == KVCA_USAGE_COUNT_MAX ? usage_count : static_cast<int32_t>(usage_count + 1);
    return pack_slot_meta(pin_count, incremented);
}

extern "C" __global__ __aicore__ void adapter_inspect_load_requests_entry(
    GM_ADDR logical_to_physical_addr,
    GM_ADDR slot_meta_addr,
    GM_ADDR logical_block_ids_addr,
    GM_ADDR current_physical_out_addr,
    GM_ADDR resident_mask_out_addr,
    GM_ADDR updated_pin_counts_out_addr,
    GM_ADDR updated_usage_counts_out_addr,
    int32_t num_logical_ids,
    int32_t block_dim) {
    __gm__ const int64_t *logical_to_physical = reinterpret_cast<__gm__ const int64_t *>(logical_to_physical_addr);
    __gm__ const kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ const kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *logical_block_ids = reinterpret_cast<__gm__ const int64_t *>(logical_block_ids_addr);
    __gm__ int64_t *current_physical_out = reinterpret_cast<__gm__ int64_t *>(current_physical_out_addr);
    __gm__ bool *resident_mask_out = reinterpret_cast<__gm__ bool *>(resident_mask_out_addr);
    __gm__ int64_t *updated_pin_counts_out = reinterpret_cast<__gm__ int64_t *>(updated_pin_counts_out_addr);
    __gm__ kvca_slotmeta_t *updated_usage_counts_out =
        reinterpret_cast<__gm__ kvca_slotmeta_t *>(updated_usage_counts_out_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t begin = chunk_begin(num_logical_ids, core_index, block_dim);
    const int32_t end = chunk_end(num_logical_ids, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        const int64_t logical_block_id = logical_block_ids[index];
        const int64_t physical_slot_id = logical_to_physical[logical_block_id];
        current_physical_out[index] = physical_slot_id;
        const bool resident = physical_slot_id >= 0;
        resident_mask_out[index] = resident;
        if (resident) {
            const kvca_slotmeta_t meta = slot_meta[physical_slot_id];
            updated_pin_counts_out[index] = unpack_pin_count(meta) + 1;
            updated_usage_counts_out[index] = static_cast<kvca_slotmeta_t>(unpack_usage_count(saturating_increment_usage(meta)));
        } else {
            updated_pin_counts_out[index] = 0;
            updated_usage_counts_out[index] = 0;
        }
    }
}

extern "C" __global__ __aicore__ void adapter_inspect_save_requests_entry(
    GM_ADDR logical_to_physical_addr,
    GM_ADDR slot_meta_addr,
    GM_ADDR logical_block_ids_addr,
    GM_ADDR current_physical_out_addr,
    GM_ADDR existing_mask_out_addr,
    GM_ADDR final_usage_counts_out_addr,
    int32_t num_logical_ids,
    int32_t block_dim) {
    __gm__ const int64_t *logical_to_physical = reinterpret_cast<__gm__ const int64_t *>(logical_to_physical_addr);
    __gm__ const kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ const kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *logical_block_ids = reinterpret_cast<__gm__ const int64_t *>(logical_block_ids_addr);
    __gm__ int64_t *current_physical_out = reinterpret_cast<__gm__ int64_t *>(current_physical_out_addr);
    __gm__ bool *existing_mask_out = reinterpret_cast<__gm__ bool *>(existing_mask_out_addr);
    __gm__ kvca_slotmeta_t *final_usage_counts_out =
        reinterpret_cast<__gm__ kvca_slotmeta_t *>(final_usage_counts_out_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t begin = chunk_begin(num_logical_ids, core_index, block_dim);
    const int32_t end = chunk_end(num_logical_ids, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        const int64_t logical_block_id = logical_block_ids[index];
        const int64_t physical_slot_id = logical_to_physical[logical_block_id];
        current_physical_out[index] = physical_slot_id;
        const bool existing = physical_slot_id >= 0;
        existing_mask_out[index] = existing;
        final_usage_counts_out[index] = existing
            ? static_cast<kvca_slotmeta_t>(unpack_usage_count(saturating_increment_usage(slot_meta[physical_slot_id])))
            : static_cast<kvca_slotmeta_t>(1);
    }
}

extern "C" __global__ __aicore__ void adapter_mark_blocked_slots_entry(
    GM_ADDR blocked_slot_ids_addr,
    GM_ADDR blocked_mask_addr,
    int32_t num_blocked_slot_ids,
    int32_t block_dim) {
    __gm__ const int64_t *blocked_slot_ids = reinterpret_cast<__gm__ const int64_t *>(blocked_slot_ids_addr);
    __gm__ bool *blocked_mask = reinterpret_cast<__gm__ bool *>(blocked_mask_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t begin = chunk_begin(num_blocked_slot_ids, core_index, block_dim);
    const int32_t end = chunk_end(num_blocked_slot_ids, core_index, block_dim);
    if (begin >= end) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    pipe.InitBuffer(calc_buf, ceil_32_bytes(sizeof(int64_t) * kMetaTileElems));
    auto blocked_ids_local = calc_buf.GetWithOffset<int64_t>(kMetaTileElems, 0);
    __gm__ uint8_t *blocked_mask_u8 = reinterpret_cast<__gm__ uint8_t *>(blocked_mask);

    for (int32_t tile_begin = begin; tile_begin < end; tile_begin += kMetaTileElems) {
        const int32_t tile_len = end - tile_begin > kMetaTileElems ? kMetaTileElems : (end - tile_begin);
        load_tile(blocked_ids_local, blocked_slot_ids + tile_begin, tile_len);
        for (int32_t inner_index = 0; inner_index < tile_len; ++inner_index) {
            blocked_mask_u8[blocked_ids_local(inner_index)] = static_cast<uint8_t>(1);
        }
    }
}

extern "C" __global__ __aicore__ void adapter_count_threshold_slots_entry(
    GM_ADDR slot_meta_addr,
    GM_ADDR blocked_mask_addr,
    GM_ADDR search_start_addr,
    GM_ADDR selection_state_addr,
    GM_ADDR local_count_workspace_addr,
    int32_t num_actual_blocks,
    int32_t threshold,
    int32_t block_dim) {
    __gm__ const kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ const kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const bool *blocked_mask = reinterpret_cast<__gm__ const bool *>(blocked_mask_addr);
    __gm__ const int64_t *search_start = reinterpret_cast<__gm__ const int64_t *>(search_start_addr);
    __gm__ const int64_t *selection_state = reinterpret_cast<__gm__ const int64_t *>(selection_state_addr);
    __gm__ int64_t *local_count_workspace = reinterpret_cast<__gm__ int64_t *>(local_count_workspace_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (selection_state[1] >= 0) {
        local_count_workspace[core_index] = 0;
        return;
    }

    const int32_t begin = chunk_begin(num_actual_blocks, core_index, block_dim);
    const int32_t end = chunk_end(num_actual_blocks, core_index, block_dim);
    const int64_t start = search_start[0];
    __gm__ const uint8_t *blocked_mask_u8 = reinterpret_cast<__gm__ const uint8_t *>(blocked_mask);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    uint32_t buffer_size =
        ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems) +
        ceil_32_bytes(sizeof(uint8_t) * kMetaTileElems);
    pipe.InitBuffer(calc_buf, buffer_size);
    int32_t offset = 0;
    auto meta_local = calc_buf.GetWithOffset<kvca_slotmeta_t>(kMetaTileElems, offset);
    offset += ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems);
    auto blocked_local = calc_buf.GetWithOffset<uint8_t>(kMetaTileElems, offset);

    int64_t local_count = 0;
    int32_t rotated_pos = begin;
    while (rotated_pos < end) {
        const int32_t actual_begin = static_cast<int32_t>((start + rotated_pos) % num_actual_blocks);
        const int32_t contiguous_len = ((end - rotated_pos) < (num_actual_blocks - actual_begin))
            ? (end - rotated_pos)
            : (num_actual_blocks - actual_begin);
        for (int32_t tile_offset = 0; tile_offset < contiguous_len; tile_offset += kMetaTileElems) {
            const int32_t tile_len =
                contiguous_len - tile_offset > kMetaTileElems ? kMetaTileElems : (contiguous_len - tile_offset);
            const int32_t tile_begin = actual_begin + tile_offset;
            load_tile(meta_local, slot_meta + tile_begin, tile_len);
            load_tile(blocked_local, blocked_mask_u8 + tile_begin, tile_len);
            for (int32_t inner_index = 0; inner_index < tile_len; ++inner_index) {
                const kvca_slotmeta_t meta = meta_local(inner_index);
                if (blocked_local(inner_index) == 0 &&
                    unpack_pin_count(meta) == 0 &&
                    unpack_usage_count(meta) == threshold) {
                    local_count += 1;
                }
            }
        }
        rotated_pos += contiguous_len;
    }
    local_count_workspace[core_index] = local_count;
}

extern "C" __global__ __aicore__ void adapter_plan_threshold_slots_entry(
    GM_ADDR local_count_workspace_addr,
    GM_ADDR local_offset_workspace_addr,
    GM_ADDR local_emit_workspace_addr,
    GM_ADDR selection_state_addr,
    int32_t block_dim,
    int32_t count,
    int32_t threshold) {
    __gm__ const int64_t *local_count_workspace =
        reinterpret_cast<__gm__ const int64_t *>(local_count_workspace_addr);
    __gm__ int64_t *local_offset_workspace = reinterpret_cast<__gm__ int64_t *>(local_offset_workspace_addr);
    __gm__ int64_t *local_emit_workspace = reinterpret_cast<__gm__ int64_t *>(local_emit_workspace_addr);
    __gm__ int64_t *selection_state = reinterpret_cast<__gm__ int64_t *>(selection_state_addr);
    if (AscendC::GetBlockIdx() != 0 || selection_state[1] >= 0) {
        return;
    }

    int64_t selected_count = selection_state[0];
    int64_t remaining = static_cast<int64_t>(count) - selected_count;
    int64_t emitted = 0;
    for (int32_t core_index = 0; core_index < block_dim; ++core_index) {
        local_offset_workspace[core_index] = selected_count + emitted;
        const int64_t available = local_count_workspace[core_index];
        int64_t emit = available;
        const int64_t still_needed = remaining - emitted;
        if (still_needed <= 0) {
            emit = 0;
        } else if (emit > still_needed) {
            emit = still_needed;
        }
        local_emit_workspace[core_index] = emit;
        emitted += emit;
    }
    selection_state[0] = selected_count + emitted;
    if (selection_state[0] == count) {
        selection_state[1] = threshold;
    }
}

extern "C" __global__ __aicore__ void adapter_collect_threshold_slots_entry(
    GM_ADDR slot_meta_addr,
    GM_ADDR blocked_mask_addr,
    GM_ADDR search_start_addr,
    GM_ADDR selection_state_addr,
    GM_ADDR local_offset_workspace_addr,
    GM_ADDR local_emit_workspace_addr,
    GM_ADDR selected_slot_ids_out_addr,
    int32_t num_actual_blocks,
    int32_t threshold,
    int32_t block_dim) {
    __gm__ const kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ const kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const bool *blocked_mask = reinterpret_cast<__gm__ const bool *>(blocked_mask_addr);
    __gm__ const int64_t *search_start = reinterpret_cast<__gm__ const int64_t *>(search_start_addr);
    __gm__ const int64_t *selection_state = reinterpret_cast<__gm__ const int64_t *>(selection_state_addr);
    __gm__ const int64_t *local_offset_workspace =
        reinterpret_cast<__gm__ const int64_t *>(local_offset_workspace_addr);
    __gm__ const int64_t *local_emit_workspace =
        reinterpret_cast<__gm__ const int64_t *>(local_emit_workspace_addr);
    __gm__ int64_t *selected_slot_ids_out = reinterpret_cast<__gm__ int64_t *>(selected_slot_ids_out_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    if (selection_state[1] >= 0 && selection_state[1] != threshold) {
        return;
    }
    const int64_t emit_count = local_emit_workspace[core_index];
    if (emit_count <= 0) {
        return;
    }

    const int32_t begin = chunk_begin(num_actual_blocks, core_index, block_dim);
    const int32_t end = chunk_end(num_actual_blocks, core_index, block_dim);
    const int64_t start = search_start[0];
    __gm__ const uint8_t *blocked_mask_u8 = reinterpret_cast<__gm__ const uint8_t *>(blocked_mask);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    uint32_t buffer_size =
        ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems) +
        ceil_32_bytes(sizeof(uint8_t) * kMetaTileElems);
    pipe.InitBuffer(calc_buf, buffer_size);
    int32_t offset = 0;
    auto meta_local = calc_buf.GetWithOffset<kvca_slotmeta_t>(kMetaTileElems, offset);
    offset += ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems);
    auto blocked_local = calc_buf.GetWithOffset<uint8_t>(kMetaTileElems, offset);

    int64_t written = 0;
    const int64_t write_offset = local_offset_workspace[core_index];
    int32_t rotated_pos = begin;
    while (rotated_pos < end && written < emit_count) {
        const int32_t actual_begin = static_cast<int32_t>((start + rotated_pos) % num_actual_blocks);
        const int32_t contiguous_len = ((end - rotated_pos) < (num_actual_blocks - actual_begin))
            ? (end - rotated_pos)
            : (num_actual_blocks - actual_begin);
        for (int32_t tile_offset = 0; tile_offset < contiguous_len && written < emit_count; tile_offset += kMetaTileElems) {
            const int32_t tile_len =
                contiguous_len - tile_offset > kMetaTileElems ? kMetaTileElems : (contiguous_len - tile_offset);
            const int32_t tile_begin = actual_begin + tile_offset;
            load_tile(meta_local, slot_meta + tile_begin, tile_len);
            load_tile(blocked_local, blocked_mask_u8 + tile_begin, tile_len);
            for (int32_t inner_index = 0; inner_index < tile_len && written < emit_count; ++inner_index) {
                const kvca_slotmeta_t meta = meta_local(inner_index);
                if (blocked_local(inner_index) == 0 &&
                    unpack_pin_count(meta) == 0 &&
                    unpack_usage_count(meta) == threshold) {
                    selected_slot_ids_out[write_offset + written] = static_cast<int64_t>(tile_begin + inner_index);
                    written += 1;
                }
            }
        }
        rotated_pos += contiguous_len;
    }
}

extern "C" __global__ __aicore__ void adapter_age_usage_entry(
    GM_ADDR slot_meta_addr,
    GM_ADDR selection_state_addr,
    int32_t num_actual_blocks,
    int32_t block_dim) {
    __gm__ kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *selection_state = reinterpret_cast<__gm__ const int64_t *>(selection_state_addr);
    const int32_t threshold = static_cast<int32_t>(selection_state[1]);
    if (threshold <= 0) {
        return;
    }

    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t begin = chunk_begin(num_actual_blocks, core_index, block_dim);
    const int32_t end = chunk_end(num_actual_blocks, core_index, block_dim);
    if (begin >= end) {
        return;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    pipe.InitBuffer(calc_buf, ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems));
    auto meta_local = calc_buf.GetWithOffset<kvca_slotmeta_t>(kMetaTileElems, 0);

    for (int32_t tile_begin = begin; tile_begin < end; tile_begin += kMetaTileElems) {
        const int32_t tile_len = end - tile_begin > kMetaTileElems ? kMetaTileElems : (end - tile_begin);
        load_tile(meta_local, slot_meta + tile_begin, tile_len);
        for (int32_t inner_index = 0; inner_index < tile_len; ++inner_index) {
            const kvca_slotmeta_t meta = meta_local(inner_index);
            const int32_t pin_count = unpack_pin_count(meta);
            const int32_t usage_count = unpack_usage_count(meta);
            meta_local.SetValue(
                inner_index,
                pack_slot_meta(pin_count, usage_count > threshold ? usage_count - threshold : 0));
        }
        store_tile(slot_meta + tile_begin, meta_local, tile_len);
    }
}

extern "C" __global__ __aicore__ void adapter_finalize_selected_slots_entry(
    GM_ADDR selection_state_addr,
    GM_ADDR search_start_addr,
    GM_ADDR selected_slot_ids_out_addr,
    int32_t num_actual_blocks,
    int32_t count) {
    __gm__ const int64_t *selection_state = reinterpret_cast<__gm__ const int64_t *>(selection_state_addr);
    __gm__ int64_t *search_start = reinterpret_cast<__gm__ int64_t *>(search_start_addr);
    __gm__ int64_t *selected_slot_ids_out = reinterpret_cast<__gm__ int64_t *>(selected_slot_ids_out_addr);
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    const int64_t selected_count = selection_state[0];
    for (int32_t index = static_cast<int32_t>(selected_count); index < count; ++index) {
        selected_slot_ids_out[index] = -1;
    }
    if (selected_count == count && count > 0) {
        search_start[0] = (selected_slot_ids_out[count - 1] + 1) % num_actual_blocks;
    }
}

extern "C" __global__ __aicore__ void adapter_commit_load_metadata_entry(
    GM_ADDR logical_to_physical_addr,
    GM_ADDR physical_to_logical_addr,
    GM_ADDR slot_meta_addr,
    GM_ADDR evicted_logical_block_ids_addr,
    int32_t num_evicted,
    GM_ADDR miss_logical_block_ids_addr,
    GM_ADDR miss_physical_slot_ids_addr,
    GM_ADDR miss_usage_counts_addr,
    int32_t num_misses,
    GM_ADDR hit_slot_ids_addr,
    GM_ADDR hit_pin_counts_addr,
    GM_ADDR hit_usage_counts_addr,
    int32_t num_hits,
    int32_t block_dim) {
    __gm__ int64_t *logical_to_physical = reinterpret_cast<__gm__ int64_t *>(logical_to_physical_addr);
    __gm__ int64_t *physical_to_logical = reinterpret_cast<__gm__ int64_t *>(physical_to_logical_addr);
    __gm__ kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *evicted_logical_block_ids =
        reinterpret_cast<__gm__ const int64_t *>(evicted_logical_block_ids_addr);
    __gm__ const int64_t *miss_logical_block_ids =
        reinterpret_cast<__gm__ const int64_t *>(miss_logical_block_ids_addr);
    __gm__ const int64_t *miss_physical_slot_ids =
        reinterpret_cast<__gm__ const int64_t *>(miss_physical_slot_ids_addr);
    __gm__ const kvca_slotmeta_t *miss_usage_counts =
        reinterpret_cast<__gm__ const kvca_slotmeta_t *>(miss_usage_counts_addr);
    __gm__ const int64_t *hit_slot_ids = reinterpret_cast<__gm__ const int64_t *>(hit_slot_ids_addr);
    __gm__ const int64_t *hit_pin_counts = reinterpret_cast<__gm__ const int64_t *>(hit_pin_counts_addr);
    __gm__ const kvca_slotmeta_t *hit_usage_counts =
        reinterpret_cast<__gm__ const kvca_slotmeta_t *>(hit_usage_counts_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());

    int32_t begin = chunk_begin(num_evicted, core_index, block_dim);
    int32_t end = chunk_end(num_evicted, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        logical_to_physical[evicted_logical_block_ids[index]] = -1;
    }

    begin = chunk_begin(num_misses, core_index, block_dim);
    end = chunk_end(num_misses, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        const int64_t logical_block_id = miss_logical_block_ids[index];
        const int64_t physical_slot_id = miss_physical_slot_ids[index];
        logical_to_physical[logical_block_id] = physical_slot_id;
        physical_to_logical[physical_slot_id] = logical_block_id;
        slot_meta[physical_slot_id] = pack_slot_meta(1, static_cast<int32_t>(miss_usage_counts[index]));
    }

    begin = chunk_begin(num_hits, core_index, block_dim);
    end = chunk_end(num_hits, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        slot_meta[hit_slot_ids[index]] = pack_slot_meta(
            static_cast<int32_t>(hit_pin_counts[index]),
            static_cast<int32_t>(hit_usage_counts[index]));
    }
}

extern "C" __global__ __aicore__ void adapter_commit_save_metadata_entry(
    GM_ADDR logical_to_physical_addr,
    GM_ADDR physical_to_logical_addr,
    GM_ADDR slot_meta_addr,
    GM_ADDR evicted_logical_block_ids_addr,
    int32_t num_evicted,
    GM_ADDR logical_block_ids_addr,
    GM_ADDR physical_slot_ids_addr,
    GM_ADDR final_pin_counts_addr,
    GM_ADDR final_usage_counts_addr,
    int32_t num_slots,
    int32_t block_dim) {
    __gm__ int64_t *logical_to_physical = reinterpret_cast<__gm__ int64_t *>(logical_to_physical_addr);
    __gm__ int64_t *physical_to_logical = reinterpret_cast<__gm__ int64_t *>(physical_to_logical_addr);
    __gm__ kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *evicted_logical_block_ids =
        reinterpret_cast<__gm__ const int64_t *>(evicted_logical_block_ids_addr);
    __gm__ const int64_t *logical_block_ids = reinterpret_cast<__gm__ const int64_t *>(logical_block_ids_addr);
    __gm__ const int64_t *physical_slot_ids = reinterpret_cast<__gm__ const int64_t *>(physical_slot_ids_addr);
    __gm__ const int64_t *final_pin_counts = reinterpret_cast<__gm__ const int64_t *>(final_pin_counts_addr);
    __gm__ const kvca_slotmeta_t *final_usage_counts =
        reinterpret_cast<__gm__ const kvca_slotmeta_t *>(final_usage_counts_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());

    int32_t begin = chunk_begin(num_evicted, core_index, block_dim);
    int32_t end = chunk_end(num_evicted, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        logical_to_physical[evicted_logical_block_ids[index]] = -1;
    }

    begin = chunk_begin(num_slots, core_index, block_dim);
    end = chunk_end(num_slots, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        const int64_t logical_block_id = logical_block_ids[index];
        const int64_t physical_slot_id = physical_slot_ids[index];
        logical_to_physical[logical_block_id] = physical_slot_id;
        physical_to_logical[physical_slot_id] = logical_block_id;
        slot_meta[physical_slot_id] = pack_slot_meta(
            static_cast<int32_t>(final_pin_counts[index]),
            static_cast<int32_t>(final_usage_counts[index]));
    }
}

extern "C" __global__ __aicore__ void adapter_release_metadata_entry(
    GM_ADDR logical_to_physical_addr,
    GM_ADDR slot_meta_addr,
    GM_ADDR logical_block_ids_addr,
    int32_t num_logical_ids,
    int32_t block_dim) {
    __gm__ const int64_t *logical_to_physical = reinterpret_cast<__gm__ const int64_t *>(logical_to_physical_addr);
    __gm__ kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ const int64_t *logical_block_ids = reinterpret_cast<__gm__ const int64_t *>(logical_block_ids_addr);
    const int32_t core_index = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t begin = chunk_begin(num_logical_ids, core_index, block_dim);
    const int32_t end = chunk_end(num_logical_ids, core_index, block_dim);
    for (int32_t index = begin; index < end; ++index) {
        const int64_t physical_slot_id = logical_to_physical[logical_block_ids[index]];
        const kvca_slotmeta_t meta = slot_meta[physical_slot_id];
        slot_meta[physical_slot_id] = pack_slot_meta(unpack_pin_count(meta) - 1, unpack_usage_count(meta));
    }
}

}  // namespace

namespace kvcache_ops {

void adapter_inspect_load_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    bool *resident_mask_out,
    int64_t *updated_pin_counts_out,
    kvca_slotmeta_t *updated_usage_counts_out,
    int32_t num_logical_ids) {
    adapter_inspect_load_requests_entry<<<block_dim, nullptr, stream>>>(
        logical_to_physical,
        slot_meta,
        logical_block_ids,
        current_physical_out,
        resident_mask_out,
        updated_pin_counts_out,
        updated_usage_counts_out,
        num_logical_ids,
        static_cast<int32_t>(block_dim));
}

void adapter_inspect_save_requests_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    const kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int64_t *current_physical_out,
    bool *existing_mask_out,
    kvca_slotmeta_t *final_usage_counts_out,
    int32_t num_logical_ids) {
    adapter_inspect_save_requests_entry<<<block_dim, nullptr, stream>>>(
        logical_to_physical,
        slot_meta,
        logical_block_ids,
        current_physical_out,
        existing_mask_out,
        final_usage_counts_out,
        num_logical_ids,
        static_cast<int32_t>(block_dim));
}

void adapter_pop_reusable_slots_kernel(
    uint32_t block_dim,
    void *stream,
    kvca_slotmeta_t *slot_meta,
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
    int32_t count) {
    if (num_blocked_slot_ids > 0) {
        adapter_mark_blocked_slots_entry<<<block_dim, nullptr, stream>>>(
            blocked_slot_ids,
            blocked_mask,
            num_blocked_slot_ids,
            static_cast<int32_t>(block_dim));
    }
    for (int32_t threshold = 0; threshold <= KVCA_USAGE_COUNT_MAX; ++threshold) {
        adapter_count_threshold_slots_entry<<<block_dim, nullptr, stream>>>(
            slot_meta,
            blocked_mask,
            search_start,
            selection_state,
            local_count_workspace,
            num_actual_blocks,
            threshold,
            static_cast<int32_t>(block_dim));
        adapter_plan_threshold_slots_entry<<<1, nullptr, stream>>>(
            local_count_workspace,
            local_offset_workspace,
            local_emit_workspace,
            selection_state,
            static_cast<int32_t>(block_dim),
            count,
            threshold);
        adapter_collect_threshold_slots_entry<<<block_dim, nullptr, stream>>>(
            slot_meta,
            blocked_mask,
            search_start,
            selection_state,
            local_offset_workspace,
            local_emit_workspace,
            selected_slot_ids_out,
            num_actual_blocks,
            threshold,
            static_cast<int32_t>(block_dim));
    }
    adapter_age_usage_entry<<<block_dim, nullptr, stream>>>(
        slot_meta,
        selection_state,
        num_actual_blocks,
        static_cast<int32_t>(block_dim));
    adapter_finalize_selected_slots_entry<<<1, nullptr, stream>>>(
        selection_state,
        search_start,
        selected_slot_ids_out,
        num_actual_blocks,
        count);
}

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
    int32_t num_hits) {
    adapter_commit_load_metadata_entry<<<block_dim, nullptr, stream>>>(
        logical_to_physical,
        physical_to_logical,
        slot_meta,
        evicted_logical_block_ids,
        num_evicted,
        miss_logical_block_ids,
        miss_physical_slot_ids,
        miss_usage_counts,
        num_misses,
        hit_slot_ids,
        hit_pin_counts,
        hit_usage_counts,
        num_hits,
        static_cast<int32_t>(block_dim));
}

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
    const int64_t *final_pin_counts,
    const kvca_slotmeta_t *final_usage_counts,
    int32_t num_slots) {
    adapter_commit_save_metadata_entry<<<block_dim, nullptr, stream>>>(
        logical_to_physical,
        physical_to_logical,
        slot_meta,
        evicted_logical_block_ids,
        num_evicted,
        logical_block_ids,
        physical_slot_ids,
        final_pin_counts,
        final_usage_counts,
        num_slots,
        static_cast<int32_t>(block_dim));
}

void adapter_release_metadata_kernel(
    uint32_t block_dim,
    void *stream,
    const int64_t *logical_to_physical,
    kvca_slotmeta_t *slot_meta,
    const int64_t *logical_block_ids,
    int32_t num_logical_ids) {
    adapter_release_metadata_entry<<<block_dim, nullptr, stream>>>(
        logical_to_physical,
        slot_meta,
        logical_block_ids,
        num_logical_ids,
        static_cast<int32_t>(block_dim));
}

}  // namespace kvcache_ops
