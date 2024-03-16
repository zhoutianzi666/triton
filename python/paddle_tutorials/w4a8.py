import triton
import triton.language as tl

import paddle
paddle.set_device('gpu')
paddle.seed(123)
from paddle.nn.quant import weight_only_linear

# @triton.autotune(
# 	configs=[
# 		triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
#  ],
# 	key=['M', 'N', 'K'],
#     reset_to_zero=['c_ptr']
# )
@triton.jit
def w4a8_kernel(
    a_ptr, b_ptr, c_ptr,
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

    # [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # [BLOCK_K, BLOCK_N] but repeated 8 times in N
    ele_per_b_dtype = 8

    #offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    b_ptrs = b_ptr + (offs_k[:, None] // ele_per_b_dtype) * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        b_shift_bits = (offs_k[:, None] % ele_per_b_dtype) * 4
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        int_b = ((b >> b_shift_bits) << 4) & 0xF0
        b = (int_b).to(tl.int8)
        #tl.device_print("kk", 4356000000000, 78909000000)
        #b = b.to(tl.int8)
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K * SPLIT_K * stride_bk // ele_per_b_dtype)  # assert BLOCK_SIZE_K % 8 == 0
    
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def w4a8(x, qweight):
    M, K = x.shape
    N = qweight.shape[1]
    output = paddle.zeros((M, N), dtype=paddle.int32) 
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )

    w4a8_kernel[grid](
        x, qweight, output,
        M, N, K,
        x.shape[1], 1,
        qweight.shape[1], 1,
        output.shape[1], 1)
    return output


if __name__ == '__main__':
    M = 4
    K = 4096*4
    N = 4096
    activation = (paddle.randn((M, K), dtype=paddle.float32) * 100).astype("int8")


    # 下面是triton的计算代码

    qweight = (paddle.randn((K // 8, N), dtype=paddle.float32) * 100 - 2).astype("int32")

    for i in range(100):
        triton_output = w4a8(activation, qweight)


    ## baseline computed by paddle op.

    qweight = qweight.numpy()
    unpack_qweight = paddle.zeros((K, N), dtype=paddle.int8).numpy()
    for i in range(K):
        for j in range(N):
            int4_id = i % 8
            int32 = qweight[i // 8, j]
            int32 = int32 >> (int4_id * 4) & 0b1111
            unpack_qweight[i, j] = int32

    unpack_qweight = paddle.to_tensor(unpack_qweight)
    unpack_qweight = unpack_qweight
    paddle_out = paddle.matmul(activation, unpack_qweight)
    
    print("paddle_out", paddle.max(paddle.abs(paddle_out - triton_output)))
    # print("paddle_out", paddle_out)
    # # print("triton_output", triton_output)
