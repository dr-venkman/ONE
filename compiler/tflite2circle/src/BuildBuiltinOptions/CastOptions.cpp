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

#include "CastOptions.h"
#include "DataLookup.h"

#include <cassert>

namespace tflite2circle
{

flatbuffers::Offset<circle::CastOptions>
build_circle_CastOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_CastOptions();
  if (tflite_builtin_options == nullptr)
    return 0;

  circle::CastOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_in_data_type(
      get_circle_tensortype(tflite_builtin_options->in_data_type()));
  builtin_options_builder.add_out_data_type(
      get_circle_tensortype(tflite_builtin_options->out_data_type()));
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
