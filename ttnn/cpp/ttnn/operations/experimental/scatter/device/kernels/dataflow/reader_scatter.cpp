// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

namespace {

FORCE_INLINE void read_input_wt_tiles() {
    // for
}

FORCE_INLINE void read_index_wt_tiles() {
    //
}

FORCE_INLINE void read_src_wt_tiles() {
    //
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const auto input_addr_gtor{make_addr_gtor<ctas.input_tensor_is_dram>(ctas.input_tensor_cb, ctas.input_tensor_addr)};
    const auto index_addr_gtor{make_addr_gtor<ctas.index_tensor_is_dram>(ctas.index_tensor_cb, ctas.index_tensor_addr)};
    const auto src_addr_gtor{make_addr_gtor<ctas.src_tensor_is_dram>(ctas.src_tensor_cb, ctas.src_tensor_addr)};

    for (uint32_t core_loop = 0; core_loop < ctas.core_loop_count; core_loop++) {
        // const uint32_t h = core_loop * total_number_of_cores +
        //                    get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        const uint32_t h =
            calculate_ht_offset_for_core(core_loop, ctas.total_number_of_cores, ctas.compute_with_storage_grid_size_x);

        read_input_wt_tiles();
        read_index_wt_tiles();
        read_src_wt_tiles();

        // for (uint32_t w = 0; w < Wt_input; w++) {
        //     cb_reserve_back(input_tensor_cb_index, one_tile);
        //     const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
        //     noc_async_read_tile(h * Wt_input + w, input_tensor_dram, l1_write_addr);
        //     noc_async_read_barrier();
        //     cb_push_back(input_tensor_cb_index, one_tile);
        // }

        // for (uint32_t w = 0; w < Wt_index; w++) {
        //     cb_wait_front(output_tensor_cb_index, one_tile);
        //     const uint32_t l1_write_addr_output = get_read_ptr(output_tensor_cb_index);
        //     noc_async_write_tile(h * Wt_index + w, output_tensor_dram, l1_write_addr_output);
        //     noc_async_write_barrier();
        //     cb_pop_front(output_tensor_cb_index, one_tile);
        // }
    }
}
