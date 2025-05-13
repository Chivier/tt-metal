// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matrix_pybind.hpp"

#include "ttnn/operations/matrix/cannon/cannon_pybind.hpp"

namespace ttnn::operations::matrix {

void py_module(py::module& module) {
    bind_cannon_operation(module);
}

}  // namespace ttnn::operations::matrix
