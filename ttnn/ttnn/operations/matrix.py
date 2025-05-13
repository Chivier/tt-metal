import ttnn

# For the golden function, use the same signature as the operation
# Keep in mind that all `ttnn.Tensor`s are converted to `torch.Tensor`s
# And arguments not needed by torch can be ignored using `*args` and `**kwargs`
def golden_function(input_tensor: "torch.Tensor", *args, **kwargs):
    output_tensor:  "torch.Tensor" = ...
    return output_tensor

# TT-NN Tensors are converted to torch tensors before calling the golden function automatically
# And the outputs are converted back to TT-NN Tensors
# But in some cases you may need to preprocess the inputs and postprocess the outputs manually

# In order to preprocess the inputs manually, use the following signature
# Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
def preprocess_golden_function_inputs(args, kwargs):
    # i.e.
    ttnn_input_tensor = args[0]
    return ttnn.to_torch(ttnn_input_tensor)

# In order to postprocess the outputs manually, use the following signature
# Note that the arguments are not packed into *args and **kwargs as in the golden function!!!
def postprocess_golden_function_outputs(args, kwargs, output):
    # i.e.
    ttnn_input_tensor = args[0]
    torch_output_tensor = output[0]
    return ttnn.from_torch(torch_output_tensor, dtype=ttnn_input_tensor.dtype, device=ttnn_input_tensor.device)

ttnn.attach_golden_function(
    ttnn.cannon,
    golden_function=golden_function,
    preprocess_golden_function_inputs=preprocess_golden_function_inputs, # Optional
    postprocess_golden_function_outputs=postprocess_golden_function_outputs # Optional
)