# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
act = Int32Scalar("act", 0) # an int32_t scalar fuse_activation
i3 = Output("op3", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
model = model.Operation("MUL", i1, i2, act).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, -3, -4, -15, 6, 23, 8, -1, -2, 3, 4, 10, -6, 7, -2],
          i2: # input 1
          [-1, -2, 3, 4, -5, -6, 7, -8, 1, -2, -3, -4, -5, 6, 7, 8]}

output0 = {i3: # output 0
           [-1, -4, -9, -16, 75, -36, 161, -64, -1, 4, -9, -16, -50, -36, 49, -16]}

# Instantiate an example
Example((input0, output0))
