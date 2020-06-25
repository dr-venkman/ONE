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
#include "CompareLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Comparison.h>
using namespace nnfw::cker;
namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

namespace
{

using OpType = onert::ir::operation::Comparison::ComparisonType;
using namespace onert::backend::cpu;

template <typename T>
void compareQuant8(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
                   OpType op_type)
{
  nnfw::cker::ComparisonParams params;
  params.left_shift = 8;
  params.input1_offset = -lhs->data_offset();
  params.input2_offset = -rhs->data_offset();
  const double norm_max_scale =
      2 * std::max(std::abs(lhs->data_scale()), std::abs(rhs->data_scale()));
  const double adjusted_lhs_scale = lhs->data_scale() / norm_max_scale;
  const double adjusted_rhs_scale = rhs->data_scale() / norm_max_scale;
  QuantizeMultiplierSmallerThanOneExp(adjusted_lhs_scale, &params.input1_multiplier,
                                      &params.input1_shift);
  QuantizeMultiplierSmallerThanOneExp(adjusted_rhs_scale, &params.input2_multiplier,
                                      &params.input2_shift);
  params.is_broadcast = !HaveSameShapes(lhs, rhs);

  if (params.is_broadcast)
  {
    switch (op_type)
    {
      case OpType::Equal:
        Broadcast4DSlowEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        Broadcast4DSlowNotEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        Broadcast4DSlowGreaterWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        Broadcast4DSlowGreaterEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        Broadcast4DSlowLessWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        Broadcast4DSlowLessEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  else // if (requires_broadcast == false)
  {
    switch (op_type)
    {
      case OpType::Equal:
        EqualWithScaling(params, getExtendedTensorShape(lhs),
                         reinterpret_cast<const T *>(lhs->buffer()), getExtendedTensorShape(rhs),
                         reinterpret_cast<const T *>(rhs->buffer()), getExtendedTensorShape(output),
                         reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        NotEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        GreaterWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        GreaterEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        LessWithScaling(params, getExtendedTensorShape(lhs),
                        reinterpret_cast<const T *>(lhs->buffer()), getExtendedTensorShape(rhs),
                        reinterpret_cast<const T *>(rhs->buffer()), getExtendedTensorShape(output),
                        reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        LessEqualWithScaling(
            params, getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  return;
}

template <typename T>
void compareScalar(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
                   OpType op_type)
{
  bool requires_broadcast = !HaveSameShapes(lhs, rhs);

  if (requires_broadcast)
  {
    switch (op_type)
    {
      case OpType::Equal:
        Broadcast4DSlowEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        Broadcast4DSlowNotEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        Broadcast4DSlowGreater(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        Broadcast4DSlowGreaterEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        Broadcast4DSlowLess(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                            getExtendedTensorShape(output),
                            reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        Broadcast4DSlowLessEqual(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  else // if (requires_broadcast == false)
  {
    switch (op_type)
    {
      case OpType::Equal:
        EqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                       getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                       getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::NotEqual:
        NotEqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                          getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                          getExtendedTensorShape(output),
                          reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Greater:
        GreaterNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                         getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                         getExtendedTensorShape(output),
                         reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::GreaterEqual:
        GreaterEqualNoScaling(
            getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
            getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
            getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::Less:
        LessNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                      getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                      getExtendedTensorShape(output), reinterpret_cast<bool *>(output->buffer()));
        break;
      case OpType::LessEqual:
        LessEqualNoScaling(getExtendedTensorShape(lhs), reinterpret_cast<const T *>(lhs->buffer()),
                           getExtendedTensorShape(rhs), reinterpret_cast<const T *>(rhs->buffer()),
                           getExtendedTensorShape(output),
                           reinterpret_cast<bool *>(output->buffer()));
        break;
      default:
        throw std::runtime_error{"Invalid OpType for CompareLayer"};
    }
  }
  return;
}
} // namespace

CompareLayer::CompareLayer()
    : _lhs(nullptr), _rhs(nullptr), _output(nullptr),
      _op_type(ir::operation::Comparison::ComparisonType::Equal)
{
  // DO NOTHING
}

void CompareLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                             const OpType op_type, IPortableTensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _op_type = op_type;
  _output = output;
}

void CompareLayer::run()
{
  if (_lhs->data_type() == OperandType::FLOAT32)
  {
    compareScalar<float>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::INT32)
  {
    compareScalar<int32_t>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::BOOL8)
  {
    compareScalar<uint8_t>(_lhs, _rhs, _output, _op_type);
  }
  else if (_lhs->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    compareQuant8<uint8_t>(_lhs, _rhs, _output, _op_type);
  }
  else
  {
    throw std::runtime_error{"Compare: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
