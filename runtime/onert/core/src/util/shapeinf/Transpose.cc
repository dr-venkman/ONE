/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{
// helper function
ir::Shape inferTransposeShape(const ir::Shape &in_shape, const std::vector<int> &perm)
{
  if (static_cast<int>(perm.size()) > in_shape.rank())
  {
    throw std::runtime_error("inferTransposeShape failed, bad rank size: " +
                             std::to_string(static_cast<int>(perm.size())));
  }
  ir::Shape out_shape(static_cast<int>(perm.size()));
  for (int idx = 0; idx < static_cast<int>(perm.size()); idx++)
  {
    if (perm[idx] < 0 || perm[idx] >= static_cast<int>(perm.size()))
    {
      throw std::runtime_error("inferTransposeShape failed, bad perm value: " +
                               std::to_string(perm[idx]));
    }
    out_shape.dim(idx) = in_shape.dim(perm[idx]);
  }
  return out_shape;
}

// StaticInferer at compilation time
void StaticInferer::visit(const ir::operation::Transpose &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);
  const auto perm{op.param().perm};
  // const auto rank{op.param().rank};
  // if input is dynamic, or perm vector is not yet filled, output also becomes dynamic
  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // set output shape, based on input and params
  ir::Shape new_shape = inferTransposeShape(input.info().shape(), perm);
  output.info().shape(new_shape);
}

// DynamicInferer at execution time
void DynamicInferer::visit(const ir::operation::Transpose &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // from op, access the buffer of second input to read new shape
  auto input_ind = op.getInputs().at(ir::operation::Transpose::Input::INPUT);
  auto input_tensor = _tensor_registry->getITensor(input_ind);
  auto input_shape = getShape(input_tensor.get());

  const auto perm{op.param().perm};
  // const auto rank{op.param().rank};
  // set output shape, based on input and params
  ir::Shape new_shape = inferTransposeShape(input_shape, perm);
  setShape(output.get(), new_shape);

  _dynamic_tensor_manager->allocate(output_ind, new_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
