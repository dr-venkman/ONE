model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
axis = Parameter("axis", "TENSOR_INT32", "{1}", [2])
keepDims = False
output = Output("output", "TENSOR_FLOAT32", "{1, 2, 1}")

model = model.Operation("REDUCE_MAX", i1, axis, keepDims).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0,
           3.0, 4.0]}

output0 = {output: # output 0
          [2.0,
           4.0]}

# Instantiate an example
Example((input0, output0))
