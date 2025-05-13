// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/cannon_device_operation.hpp"

namespace ttnn::operations::matrix {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct CompositeCannonOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static Tensor invoke(const Tensor& input_tensor) {
        auto copy = prim::cannon(input_tensor);
        auto another_copy = prim::cannon(copy);
        return another_copy;
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto composite_cannon =
    ttnn::register_operation<"ttnn::composite_cannon", operations::matrix::CompositeCannonOperation>();
}  // namespace ttnn
