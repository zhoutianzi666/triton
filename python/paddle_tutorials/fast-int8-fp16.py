
import paddle

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    write_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = block_start + tl.arange(0, BLOCK_SIZE) 
    #offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # ele_per_b_dtype = 2
    # b_shift_bits = (offsets % ele_per_b_dtype) * 4
    # x = (x >> b_shift_bits).to(tl.uint16)

    magic_number = (0x00006400)
    magic_number = magic_number.to(tl.uint16)
    output = x | magic_number
    output = output.to(tl.float16, bitcast=True)
    output = output - 1152

    tl.store(output_ptr + write_offsets, output, mask=mask)



def add(x: paddle.Tensor):
    output = paddle.empty(x.shape, dtype="float16")
    n_elements = (int)(x.numel())
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, output, n_elements, BLOCK_SIZE=2048)
    return output



shape_tensor = paddle.to_tensor([2048*16], dtype=paddle.int32)
x = paddle.randn(shape_tensor, dtype=paddle.float16) * 10
x = x.astype("int8")
real_x = paddle.assign(x)
x = x.astype("int32")
x = x + 128
x = x.astype("uint8")

triton_output = add(x)

print(paddle.max(real_x.astype("float32") - triton_output.astype("float32")))
print(paddle.min(real_x.astype("float32") - triton_output.astype("float32")))


