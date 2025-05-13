// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/matrix/cannon/cannon.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::matrix {

void bind_cannon_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::prim::cannon,
        R"doc(cannon(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // The overload with `queue_id` argument will be added automatically for primitive operations
        // This specific function can be called from python as `ttnn.prim.cannon(input_tensor)` or
        // `ttnn.prim.cannon(input_tensor, queue_id=queue_id)`
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::prim::cannon)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            py::arg("input_tensor")});

    bind_registered_operation(
        module,
        ttnn::composite_cannon,
        R"doc(composite_cannon(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::composite_cannon)& self, const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                return self(input_tensor);
            },
            py::arg("input_tensor")});
}

}  // namespace ttnn::operations::matrix
