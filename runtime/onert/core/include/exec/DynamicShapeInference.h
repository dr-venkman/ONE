/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_EXEC_DYNAMIC_SHAPE_INFERENCE_H__
#define __ONERT_EXEC_DYNAMIC_SHAPE_INFERENCE_H__

#include "ir/Operands.h"
#include "ir/OperationVisitor.h"
#include "ir/Index.h"
#include "backend/IDynamicTensorManager.h"
#include "backend/ITensorManager.h"
#include "backend/ITensorRegistry.h"

#include <map>

namespace onert
{
namespace exec
{

/**
 * @brief Class to infer shape of output tensor at execution time and
 *        allocate memory fo output tensor if needed
 */
class DynamicShapeInferer : public ir::OperationVisitor
{
public:
  DynamicShapeInferer(const ir::Operands &operands, backend::IDynamicTensorManager *tensor_manager,
                      const std::shared_ptr<backend::ITensorRegistry> &tensor_registry)
      : _operands(operands), _dynamic_tensor_manager(tensor_manager),
        _tensor_registry(tensor_registry)
  {
    UNUSED_RELEASE(_operands);
    UNUSED_RELEASE(_dynamic_tensor_manager);
    UNUSED_RELEASE(_tensor_registry);
  }

public:
  // TODO Define visitors for operations. List them in alphabetic order.
  // Remove TODO when any op starting from the alphabet is added
  void visit(const ir::operation::Abs &op) override;
  void visit(const ir::operation::Add &op) override;
  void visit(const ir::operation::ArgMax &op) override;
  void visit(const ir::operation::BatchMatMul &op) override;
  void visit(const ir::operation::BroadcastTo &op) override;
  void visit(const ir::operation::Cast &op) override;
  void visit(const ir::operation::Comparison &op) override;
  void visit(const ir::operation::Concat &op) override;
  void visit(const ir::operation::Conv2D &op) override;
  void visit(const ir::operation::Cos &op) override;
  void visit(const ir::operation::Div &op) override;
  void visit(const ir::operation::Exp &op) override;
  void visit(const ir::operation::ExpandDims &op) override;
  void visit(const ir::operation::Fill &op) override;
  void visit(const ir::operation::FullyConnected &op) override;
  void visit(const ir::operation::FusedBatchNorm &op) override;
  void visit(const ir::operation::Gather &op) override;
  void visit(const ir::operation::Log &op) override;
  void visit(const ir::operation::LogicalNot &op) override;
  void visit(const ir::operation::LogicalOr &op) override;
  void visit(const ir::operation::Logistic &op) override;
  void visit(const ir::operation::MatrixBandPart &op) override;
  void visit(const ir::operation::Max &op) override;
  void visit(const ir::operation::Min &op) override;
  void visit(const ir::operation::Mul &op) override;
  void visit(const ir::operation::Neg &op) override;
  void visit(const ir::operation::OneHot &op) override;
  void visit(const ir::operation::Pack &op) override;
  void visit(const ir::operation::Pad &op) override;
  void visit(const ir::operation::Permute &op) override;
  void visit(const ir::operation::Pow &op) override;
  // TODO write op starting from Q
  void visit(const ir::operation::Range &op) override;
  void visit(const ir::operation::Reduce &op) override;
  void visit(const ir::operation::Reshape &op) override;
  void visit(const ir::operation::Round &op) override;
  void visit(const ir::operation::RSQRT &op) override;
  void visit(const ir::operation::Reverse &op) override;
  void visit(const ir::operation::Select &op) override;
  void visit(const ir::operation::Shape &op) override;
  void visit(const ir::operation::Sin &op) override;
  void visit(const ir::operation::Slice &op) override;
  void visit(const ir::operation::Softmax &op) override;
  void visit(const ir::operation::Split &op) override;
  void visit(const ir::operation::Squeeze &op) override;
  void visit(const ir::operation::StridedSlice &op) override;
  void visit(const ir::operation::Sub &op) override;
  void visit(const ir::operation::SquaredDifference &op) override;
  void visit(const ir::operation::Tanh &op) override;
  void visit(const ir::operation::Tile &op) override;
  void visit(const ir::operation::Transpose &op) override;
  void visit(const ir::operation::Unpack &op) override;
  // TODO write op starting from V
  void visit(const ir::operation::ZerosLike &op) override;

private:
  /**
   * @brief Performs shape inference and memory allocation for arithmetic operation
   */
  void handleBinaryArithmeticOp(const ir::Operation &op, const ir::OperandIndex lhs_idx,
                                const ir::OperandIndex rhs_idx);
  /**
   * @brief Performs shape inference and memory allocation for unary op whose output shape is
   *        always same with input shape
   */
  void handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_idx);

private:
  /**
   * @brief To get operand-level info, e.g., ir::Operand::isConstant()
   */
  const ir::Operands &_operands;
  /**
   * @brief To allocate memory for output tensor if needed
   */
  backend::IDynamicTensorManager *_dynamic_tensor_manager;
  /**
   * @brief To get tensor object and access tensor-level info, e.g., ITensor::buffer()
   */
  std::shared_ptr<backend::ITensorRegistry> _tensor_registry;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_DYNAMIC_SHAPE_INFERENCE_H__
