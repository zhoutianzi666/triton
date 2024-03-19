import triton
import triton.language as tl

import paddle
paddle.seed(123)
from paddle.nn.quant import weight_only_linear

# @triton.autotune(
# 	configs=[
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
# 		triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),
#         triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 512, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4),

#  ],
# 	key=['M', 'N', 'K'],
#     reset_to_zero=['c_ptr']
# )
@triton.jit
def wint4_kernel(
    a_ptr, b_ptr, c_ptr,
    bs_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr
    ):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    ele_per_b_dtype = 2
    
    # modeify_offs_k = offs_k % (BLOCK_SIZE_K // ele_per_b_dtype) + pid_sp_k * BLOCK_SIZE_K // ele_per_b_dtype
    # modeify_offs_k = tl.max_contiguous(tl.multiple_of(modeify_offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    # b_ptrs = b_ptr + modeify_offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    # old_offs_k = tl.max_contiguous(tl.multiple_of(offs_k // ele_per_b_dtype, BLOCK_SIZE_K), BLO

    b_ptrs = b_ptr + (offs_k[:,None] // ele_per_b_dtype) * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    magic_number = (0x00006400)

    #b_shift_bits = ((offs_k[:, None] % BLOCK_SIZE_K) // (BLOCK_SIZE_K // ele_per_b_dtype)) * 4
    b_shift_bits = (offs_k[:, None] % ele_per_b_dtype) * 4

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        bs_ptrs = bs_ptr + offs_bn[None, :]
        bs = tl.load(bs_ptrs)
        
        # this is naive method
        # fp_b = (b >> b_shift_bits) & 0xF
        # fp_b = fp_b - 8
        # fp_b = fp_b.to(tl.float16)

        # this is nv's method.
        fp_b = (b >> b_shift_bits & 0xF) | magic_number
        fp_b = fp_b.to(tl.uint16)
        fp_b = fp_b.to(tl.float16, bitcast=True)
        fp_b = fp_b - 1032

        fp_b = fp_b * bs
        accumulator += tl.dot(a, fp_b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // ele_per_b_dtype)


    # only let the first block do epilogue
    if bias_ptr and pid_sp_k == 0:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        accumulator += bias[None,:]

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def matmul_dequantize_int4_s2(x, 
                              qweight, 
                              scales, 
                              qzeros, 
                              group_size: int = 128, 
                              output=None):
    """
    """
    M, K = x.shape
    N = scales.shape[-1]
    N = qweight.shape[-1]
    output = paddle.zeros((M, N), dtype=paddle.float16) 
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )

    wint4_kernel[grid](
        x, qweight, output,
        scales,
        M, N, K,
        x.shape[1], 1,
        qweight.shape[1], 1,
        output.shape[1], 1,
    )
    return output


if __name__ == '__main__':
    M = 510
    K = 5120
    N = 5120
    activation = paddle.randn((M, K), dtype=paddle.float16)
    original_weight = paddle.randn((K, N), dtype=paddle.float16)
    #from paddle.nn.quant import weight_quantize
    #qweight, scale = weight_quantize(original_weight, algo="weight_only_int4")

    # 下面这个是故意让他除以2的！
    qweight = paddle.zeros((N // 2, K), dtype=paddle.int8)
    scale = paddle.randn((K,), dtype=paddle.float16)
    bias = paddle.randn((N,), dtype=paddle.float16)

    for i in range(100):
        paddle_output = weight_only_linear(activation, 
                                        qweight,
                                        bias,
                                        scale,
                                        "int4",)



    # 下面是triton的计算代码

    activation = paddle.randn((M, K), dtype=paddle.float16)
    qweight = (paddle.randn((K // 8, N), dtype=paddle.float32) * 100 - 2).astype("int32")
    scale = paddle.randn((K,), dtype=paddle.float16)

    for i in range(100):
        triton_output = matmul_dequantize_int4_s2(
            activation,
            qweight,
            scale,
            None)


    ## baseline computed by paddle op.
    qweight = qweight.numpy()
    unpack_qweight = paddle.zeros((K, N), dtype=paddle.int32).numpy()
    for i in range(K):
        for j in range(N):
            int4_id = i % 8
            int32 = qweight[i // 8, j]
            int32 = int32 >> (int4_id * 4) & 0b1111
            unpack_qweight[i, j] = int32

    unpack_qweight = paddle.to_tensor(unpack_qweight)
    unpack_qweight = unpack_qweight.cast("float16") * scale.reshape([K, 1])
    paddle_out = paddle.matmul(activation, unpack_qweight)

    print("paddle_out", paddle.max(paddle.abs(paddle_out - triton_output)))
    print("paddle_out", paddle_out)
    print("triton_output", triton_output)
