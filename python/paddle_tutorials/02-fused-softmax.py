
import paddle
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = paddle.empty((n_rows, n_cols), dtype=x.dtype)
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.shape[1],
        y.shape[1],
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

shape_tensor = paddle.to_tensor([1823, 781], dtype=paddle.int32)
x = paddle.randn(shape_tensor, dtype=paddle.float16)

triton_output = softmax(x)
paddle_output = paddle.nn.functional.softmax(x, axis=1)
if paddle.allclose(triton_output, paddle_output, atol=1e-3, rtol=0.0):
    print("✅ Triton and Paddle match")
else:
    print("❌ Triton and Paddle differ")

