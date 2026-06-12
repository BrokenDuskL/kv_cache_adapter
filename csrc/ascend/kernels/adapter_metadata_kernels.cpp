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

template <AscendC::HardEvent event>
__aicore__ inline void sync_pipe(AscendC::TPipe &pipe) {
    const int32_t event_id = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(event_id);
    AscendC::WaitFlag<event>(event_id);
}

template <typename T>
__aicore__ inline void load_tile_pad(
    const AscendC::LocalTensor<T> &local_tensor,
    const __gm__ T *global_ptr,
    int32_t len) {
    AscendC::GlobalTensor<T> global_tensor;
    global_tensor.SetGlobalBuffer(const_cast<__gm__ T *>(global_ptr), len);
    AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(local_tensor, global_tensor, copy_params, pad_params);
}

__aicore__ inline void load_int64_tile_pad(
    const AscendC::LocalTensor<int64_t> &local_tensor,
    const __gm__ int64_t *global_ptr,
    int32_t len) {
    AscendC::GlobalTensor<int64_t> global_tensor;
    global_tensor.SetGlobalBuffer(const_cast<__gm__ int64_t *>(global_ptr), len);
    AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(len * sizeof(int64_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> pad_params{false, 0, 0, 0};
    AscendC::DataCopyPad(local_tensor, global_tensor, copy_params, pad_params);
}

template <typename T>
__aicore__ inline void store_tile_pad(
    __gm__ T *global_ptr,
    const AscendC::LocalTensor<T> &local_tensor,
    int32_t len) {
    AscendC::GlobalTensor<T> global_tensor;
    global_tensor.SetGlobalBuffer(global_ptr, len);
    AscendC::DataCopyExtParams copy_params{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(global_tensor, local_tensor, copy_params);
}

template <typename T>
inline void *launch_arg(T *ptr) {
    return reinterpret_cast<void *>(ptr);
}

template <typename T>
inline void *launch_arg(const T *ptr) {
    return const_cast<void *>(reinterpret_cast<const void *>(ptr));
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
    __gm__ uint8_t *resident_mask_out = reinterpret_cast<__gm__ uint8_t *>(resident_mask_out_addr);
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
        resident_mask_out[index] = resident ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
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
    __gm__ uint8_t *existing_mask_out = reinterpret_cast<__gm__ uint8_t *>(existing_mask_out_addr);
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
        existing_mask_out[index] = existing ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
        final_usage_counts_out[index] = existing
            ? static_cast<kvca_slotmeta_t>(unpack_usage_count(saturating_increment_usage(slot_meta[physical_slot_id])))
            : static_cast<kvca_slotmeta_t>(1);
    }
}

extern "C" __global__ __aicore__ void adapter_pop_reusable_slots_entry(
    GM_ADDR slot_meta_addr,
    GM_ADDR search_start_addr,
    GM_ADDR blocked_slot_ids_addr,
    GM_ADDR selected_slot_ids_out_addr,
    int32_t num_actual_blocks,
    int32_t num_blocked_slot_ids,
    int32_t count) {
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    __gm__ kvca_slotmeta_t *slot_meta = reinterpret_cast<__gm__ kvca_slotmeta_t *>(slot_meta_addr);
    __gm__ int64_t *search_start = reinterpret_cast<__gm__ int64_t *>(search_start_addr);
    __gm__ const int64_t *blocked_slot_ids = reinterpret_cast<__gm__ const int64_t *>(blocked_slot_ids_addr);
    __gm__ int64_t *selected_slot_ids_out = reinterpret_cast<__gm__ int64_t *>(selected_slot_ids_out_addr);

    AscendC::TPipe pipe;
    AscendC::TBuf<> calc_buf;
    const uint32_t int64_buffer_size = ceil_32_bytes(sizeof(int64_t) * kMetaTileElems);
    const uint32_t meta_buffer_size = ceil_32_bytes(sizeof(kvca_slotmeta_t) * kMetaTileElems);
    pipe.InitBuffer(calc_buf, int64_buffer_size + meta_buffer_size);
    auto blocked_ids_local = calc_buf.GetWithOffset<int64_t>(kMetaTileElems, 0);
    auto meta_local = calc_buf.GetWithOffset<kvca_slotmeta_t>(kMetaTileElems, int64_buffer_size);
    const bool blocked_ids_in_local = num_blocked_slot_ids > 0 && num_blocked_slot_ids <= kMetaTileElems;
    if (blocked_ids_in_local) {
        load_int64_tile_pad(blocked_ids_local, blocked_slot_ids, num_blocked_slot_ids);
        sync_pipe<AscendC::HardEvent::MTE2_S>(pipe);
    }

    int64_t selected_count = 0;
    int64_t selected_threshold = -1;
    const int64_t start = search_start[0];
    for (int32_t threshold = 0; threshold <= KVCA_USAGE_COUNT_MAX && selected_count < count; ++threshold) {
        int32_t rotated_pos = 0;
        while (rotated_pos < num_actual_blocks && selected_count < count) {
            const int32_t actual_begin = static_cast<int32_t>((start + rotated_pos) % num_actual_blocks);
            const int32_t contiguous_len = ((num_actual_blocks - rotated_pos) < (num_actual_blocks - actual_begin))
                ? (num_actual_blocks - rotated_pos)
                : (num_actual_blocks - actual_begin);
            for (int32_t tile_offset = 0; tile_offset < contiguous_len && selected_count < count;
                 tile_offset += kMetaTileElems) {
                const int32_t tile_len = contiguous_len - tile_offset > kMetaTileElems
                    ? kMetaTileElems
                    : (contiguous_len - tile_offset);
                const int32_t tile_begin = actual_begin + tile_offset;
                load_tile_pad(meta_local, slot_meta + tile_begin, tile_len);
                sync_pipe<AscendC::HardEvent::MTE2_S>(pipe);
                for (int32_t inner_index = 0; inner_index < tile_len && selected_count < count; ++inner_index) {
                    const int64_t slot_id = static_cast<int64_t>(tile_begin + inner_index);
                    bool is_blocked = false;
                    if (blocked_ids_in_local) {
                        for (int32_t blocked_index = 0; blocked_index < num_blocked_slot_ids; ++blocked_index) {
                            if (blocked_ids_local(blocked_index) == slot_id) {
                                is_blocked = true;
                                break;
                            }
                        }
                    } else {
                        for (int32_t blocked_index = 0; blocked_index < num_blocked_slot_ids; ++blocked_index) {
                            if (blocked_slot_ids[blocked_index] == slot_id) {
                                is_blocked = true;
                                break;
                            }
                        }
                    }
                    const kvca_slotmeta_t meta = meta_local(inner_index);
                    if (!is_blocked &&
                        unpack_pin_count(meta) == 0 &&
                        unpack_usage_count(meta) == threshold) {
                        selected_slot_ids_out[selected_count] = slot_id;
                        selected_count += 1;
                    }
                }
                sync_pipe<AscendC::HardEvent::S_MTE2>(pipe);
            }
            rotated_pos += contiguous_len;
        }
        if (selected_count == count) {
            selected_threshold = threshold;
        }
    }

    if (selected_threshold > 0) {
        for (int32_t tile_begin = 0; tile_begin < num_actual_blocks; tile_begin += kMetaTileElems) {
            const int32_t tile_len =
                num_actual_blocks - tile_begin > kMetaTileElems ? kMetaTileElems : (num_actual_blocks - tile_begin);
            load_tile_pad(meta_local, slot_meta + tile_begin, tile_len);
            sync_pipe<AscendC::HardEvent::MTE2_S>(pipe);
            for (int32_t inner_index = 0; inner_index < tile_len; ++inner_index) {
                const kvca_slotmeta_t meta = meta_local(inner_index);
                const int32_t pin_count = unpack_pin_count(meta);
                const int32_t usage_count = unpack_usage_count(meta);
                meta_local.SetValue(
                    inner_index,
                    pack_slot_meta(pin_count, usage_count > selected_threshold ? usage_count - selected_threshold : 0));
            }
            sync_pipe<AscendC::HardEvent::S_MTE3>(pipe);
            store_tile_pad(slot_meta + tile_begin, meta_local, tile_len);
            sync_pipe<AscendC::HardEvent::MTE3_S>(pipe);
        }
    }

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
    __gm__ const kvca_slotmeta_t *final_pin_counts =
        reinterpret_cast<__gm__ const kvca_slotmeta_t *>(final_pin_counts_addr);
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
    uint8_t *resident_mask_out,
    int64_t *updated_pin_counts_out,
    kvca_slotmeta_t *updated_usage_counts_out,
    int32_t num_logical_ids) {
    adapter_inspect_load_requests_entry<<<block_dim, nullptr, stream>>>(
        launch_arg(logical_to_physical),
        launch_arg(slot_meta),
        launch_arg(logical_block_ids),
        launch_arg(current_physical_out),
        launch_arg(resident_mask_out),
        launch_arg(updated_pin_counts_out),
        launch_arg(updated_usage_counts_out),
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
    uint8_t *existing_mask_out,
    kvca_slotmeta_t *final_usage_counts_out,
    int32_t num_logical_ids) {
    adapter_inspect_save_requests_entry<<<block_dim, nullptr, stream>>>(
        launch_arg(logical_to_physical),
        launch_arg(slot_meta),
        launch_arg(logical_block_ids),
        launch_arg(current_physical_out),
        launch_arg(existing_mask_out),
        launch_arg(final_usage_counts_out),
        num_logical_ids,
        static_cast<int32_t>(block_dim));
}

void adapter_pop_reusable_slots_kernel(
    void *stream,
    kvca_slotmeta_t *slot_meta,
    int64_t *search_start,
    const int64_t *blocked_slot_ids,
    int64_t *selected_slot_ids_out,
    int32_t num_actual_blocks,
    int32_t num_blocked_slot_ids,
    int32_t count) {
    adapter_pop_reusable_slots_entry<<<1, nullptr, stream>>>(
        launch_arg(slot_meta),
        launch_arg(search_start),
        launch_arg(blocked_slot_ids),
        launch_arg(selected_slot_ids_out),
        num_actual_blocks,
        num_blocked_slot_ids,
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
        launch_arg(logical_to_physical),
        launch_arg(physical_to_logical),
        launch_arg(slot_meta),
        launch_arg(evicted_logical_block_ids),
        num_evicted,
        launch_arg(miss_logical_block_ids),
        launch_arg(miss_physical_slot_ids),
        launch_arg(miss_usage_counts),
        num_misses,
        launch_arg(hit_slot_ids),
        launch_arg(hit_pin_counts),
        launch_arg(hit_usage_counts),
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
    const kvca_slotmeta_t *final_pin_counts,
    const kvca_slotmeta_t *final_usage_counts,
    int32_t num_slots) {
    adapter_commit_save_metadata_entry<<<block_dim, nullptr, stream>>>(
        launch_arg(logical_to_physical),
        launch_arg(physical_to_logical),
        launch_arg(slot_meta),
        launch_arg(evicted_logical_block_ids),
        num_evicted,
        launch_arg(logical_block_ids),
        launch_arg(physical_slot_ids),
        launch_arg(final_pin_counts),
        launch_arg(final_usage_counts),
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
        launch_arg(logical_to_physical),
        launch_arg(slot_meta),
        launch_arg(logical_block_ids),
        num_logical_ids,
        static_cast<int32_t>(block_dim));
}

}  // namespace kvcache_ops
