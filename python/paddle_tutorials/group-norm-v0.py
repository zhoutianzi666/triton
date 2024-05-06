
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import triton
import triton.language as tl
import paddle
# 保证 H*W 可以整除 BLOCKSIZE 
@triton.jit
def group_norm_kernel(
    sample_ptr,  # pointer to the input
    output_ptr,
    weight_ptr,
    bias_ptr,
    eps,  # epsilon to avoid division by zero
    channels_num,
    group_size,
    channels_stride,
    group_stride,
    batch_stride,
    num_groups,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)

    group_bias = batch_id * batch_stride + group_id * group_stride
    # 要处理溢出
    _sum = 0.0
    # 求和
    sample_ptrs = sample_ptr + group_bias + tl.arange(0, BLOCK_SIZE * BLOCK_SIZE_G)
    for k in range(0, tl.cdiv(group_stride, BLOCK_SIZE * BLOCK_SIZE_G)):
        sample = tl.load(sample_ptrs, mask=tl.arange(0, BLOCK_SIZE * BLOCK_SIZE_G) < (group_stride - k * BLOCK_SIZE * BLOCK_SIZE_G), other=0.0)
        # tl.device_print("", tl.sum(sample))
        _sum = _sum + tl.sum(sample)
        sample_ptrs += BLOCK_SIZE * BLOCK_SIZE_G
    _mean = _sum / group_stride
    
    # 求方差
    _var = 0.0
    sample_ptrs = sample_ptr + group_bias + tl.arange(0, BLOCK_SIZE * BLOCK_SIZE_G)
    for k in range(0, tl.cdiv( group_stride, BLOCK_SIZE * BLOCK_SIZE_G)):
        sample = tl.load(sample_ptrs , mask=tl.arange(0, BLOCK_SIZE * BLOCK_SIZE_G) < (group_stride - k * BLOCK_SIZE * BLOCK_SIZE_G), other = _mean)
        center_bias = sample - _mean
        center_bias_value = tl.sum(center_bias * center_bias)
        _var = _var + center_bias_value
        sample_ptrs += BLOCK_SIZE * BLOCK_SIZE_G
    _var = _var / group_stride
    
    # 求归一化
    rstd = 1 / tl.sqrt(_var + eps)
    offset_channel = tl.arange(0, BLOCK_SIZE_G) 
    offset_group = tl.arange(0, BLOCK_SIZE) 
    ptrs_offset = group_bias + offset_channel[:, None] * channels_stride + offset_group[None,:]

    sample_ptrs = sample_ptr + ptrs_offset
    output_ptrs = output_ptr + ptrs_offset
    weight_para = tl.zeros((BLOCK_SIZE_G, 1), dtype=tl.float32)
    bias_para = tl.zeros((BLOCK_SIZE_G, 1), dtype=tl.float32)
    if weight_ptr: 
        weight_para = tl.load(weight_ptr + group_id * group_size + offset_channel[:, None], mask = (offset_channel[:, None] < channels_num - group_id * group_size), other = 0.0)
    if bias_ptr:
        bias_para = tl.load(bias_ptr + group_id * group_size + offset_channel[:, None], mask = (offset_channel[:, None] < channels_num - group_id * group_size), other = 0.0)
    for k in range(0, tl.cdiv(channels_stride, BLOCK_SIZE)):
        cc_mask = ((offset_group[None, :] < channels_stride - k * BLOCK_SIZE) & (offset_channel[:, None] < channels_num - group_id * group_size))
        sample = tl.load(sample_ptrs, mask = cc_mask, other = 0.0)
        re = (sample - _mean) * rstd
        if weight_ptr:
            re = re * weight_para
        if bias_ptr:
            re = re + bias_para
        tl.store(output_ptrs, re, mask = cc_mask)
        sample_ptrs += BLOCK_SIZE
        output_ptrs += BLOCK_SIZE

def group_norm(sample, num_group, eps=1e-5, weight = None , bias = None):
    N,C,H,W = sample.shape
    group_size = int((C + num_group - 1) / num_group)
    # print(group_size)
    # print(type(group_size))
    grid = lambda META: (
        N,
        num_group,
    )
    output = paddle.empty((N, C, H, W), dtype=sample.dtype)
    group_norm_kernel[grid](sample, 
                            output,
                            weight,
                            bias, 
                            eps,
                            C, 
                            group_size,
                            H*W,
                            group_size * H * W, 
                            C * H * W,
                            num_group, 
                            BLOCK_SIZE=128,
                            BLOCK_SIZE_G = triton.next_power_of_2(group_size))
    return output

import numpy as np
N,C,H,W,num_group = 1,3,10,10,1
shape_tensor_1 = paddle.to_tensor([N, C, H, W], dtype=paddle.int32)
# paddle.seed(100)
sample = paddle.randn(shape_tensor_1, dtype=paddle.float32)
sample  = sample

# weight and bias 
weight = np.array(np.random.random((C)), dtype=np.float32)
print(weight)
weight_tentor = paddle.to_tensor(weight)
bias = np.array(np.random.random((C)), dtype=np.float32)
print(bias)
bias_tentor = paddle.to_tensor(bias)
weight_paddle = paddle.ParamAttr(name= "haha", initializer=paddle.nn.initializer.Assign(weight))
bias_paddle = paddle.ParamAttr(name= "bias", initializer=paddle.nn.initializer.Assign(bias))
print("=======INPUT======")
output = group_norm(sample, num_group, 1e-5, weight_tentor, bias_tentor)
print("====TRITON===OUTPUT=====")

group_norm_paddle = paddle.nn.GroupNorm(num_channels=C, num_groups=num_group, weight_attr = weight_paddle ,bias_attr = bias_paddle, epsilon=1e-5)
print("====PADDLE===OUTPUT=====")
group_norm_out = group_norm_paddle(sample)
print(paddle.max(group_norm_out-output))



