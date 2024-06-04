import triton
import triton.language as tl
import paddle
# 保证 H*W 可以整除 BLOCKSIZE 
import os


import paddle


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def get_group_norm_kernel_config():
    configs = []
    for num_stages in [2,3,4]:
        for block_m in [32, 64, 128, 256]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                    },
                    num_stages=num_stages,
                    num_warps=4,
                )
            )
    return configs

@triton.autotune(
    configs=get_group_norm_kernel_config(),
    key=['batch_stride', 'channel_stride', 'hw_stride', 'group_stride', 'group_num'],
    reset_to_zero=['output_sum_ptr', 'output_sum_squares_ptr'],
)
@triton.jit
def group_norm_first_stage(
    sample_ptr,
    output_sum_ptr,
    output_sum_squares_ptr,
    batch_stride,
    channel_stride,
    hw_stride,
    group_stride,
    group_num,
    group_size, # numbers of channel
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
):
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_id = tl.program_id(2)
    offset_channel = tl.arange(0, BLOCK_SIZE_G)
    offset_block = tl.arange(0, BLOCK_SIZE_M)
    data_start = batch_id * batch_stride + group_id * group_stride
    sample_ptrs = sample_ptr + data_start + offset_channel[:, None] * channel_stride + offset_block[None, :] * hw_stride + block_id * BLOCK_SIZE_M
    tl.static_print("sample_ptrs", sample_ptrs.dtype)
    # 计算均值
    channel_mask = offset_channel[:,None] < group_size
    offset_block_mask = offset_block[None, :] <  (channel_stride - block_id * BLOCK_SIZE_M)
    sample_ = tl.load(sample_ptrs, mask = channel_mask & offset_block_mask, other=0.0)
    # tl.static_print(" SAMPLE TYPE ", sample.dtype)
    sample = sample_.to(tl.float32)
    # sample_fp32 = sample
    _sum = tl.sum(sample)
    
    _sum_squares = tl.sum(sample * sample)
    # tl.static_print(_sum)
    # tl.static_print(_sum_squares)
    # # 直接add
    output_start = batch_id * group_num + group_id
    tl.atomic_add(output_sum_ptr + output_start, _sum)
    tl.atomic_add(output_sum_squares_ptr + output_start, _sum_squares)

@triton.autotune(
    configs=get_group_norm_kernel_config(),
    key=['batch_stride', 'channel_stride', 'hw_stride', 'group_stride', 'group_num'],
)
@triton.jit
def group_norm_second_stage(
    sample_ptr,
    output_ptr,
    output_sum_ptr,
    output_sum_squares_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    batch_stride,
    channel_stride,
    hw_stride,
    group_stride,
    group_num,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_G: tl.constexpr,
):
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_id = tl.program_id(2)
    offset_channel = (tl.arange(0, BLOCK_SIZE_G)) % group_size
    offset_block = tl.arange(0, BLOCK_SIZE_M)
    data_start = batch_id * batch_stride + group_id * group_stride
    sample_ptrs = sample_ptr + data_start + offset_channel[:, None] * channel_stride + (offset_block[None,:] * hw_stride + block_id * BLOCK_SIZE_M) % channel_stride
    sample_ = tl.load(sample_ptrs)
    sample = sample_.to(tl.float32)
    # 
    start = batch_id * group_num + group_id
    # 计算均值
    _sum = tl.load(output_sum_ptr + start)
    # tl.device_print("sum",_sum)
    _mean = _sum / group_stride
    # 计算方差
    _sum_squares = tl.load(output_sum_squares_ptr + start)
    _var = _sum_squares / group_stride - _mean * _mean
    rstd = 1 / tl.sqrt(_var + eps)
    
    weight_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    bias_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    weight_para_temp = tl.load(weight_ptr + group_id * group_size + offset_channel[:,None])
    weight_para = weight_para_temp.to(tl.float32)
    bias_para_temp = tl.load(bias_ptr + group_id * group_size + offset_channel[:,None])
    bias_para = bias_para_temp.to(tl.float32)

    re_ = (sample - _mean) * rstd
    re_ = re_ * weight_para
    re_ = re_ + bias_para
    re = re_.to(tl.float16)
    output_ptrs = output_ptr + data_start + offset_channel[:, None] * channel_stride + (offset_block[None,:] * hw_stride + block_id * BLOCK_SIZE_M) % (channel_stride)
    tl.store(output_ptrs, re)

def group_norm(sample, num_group, eps=1e-5, weight = None , bias = None):
    N,C,H,W = sample.shape
    group_size = int((C + num_group - 1) / num_group)
    # print(group_size)
    # print(type(group_size))
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_G = triton.next_power_of_2(group_size)

    grid = lambda META: (
        N,
        num_group,
        triton.cdiv(H*W, META['BLOCK_SIZE_M'])
    )
    # print(sample.dtype)
    output = paddle.empty((N, C, H, W), dtype=sample.dtype)
    output_sum = paddle.zeros((N, num_group), dtype=paddle.float32)
    output_sum_squares = paddle.zeros((N, num_group), dtype=paddle.float32)
    batch_stride, channel_stride, group_stride = C * H * W, H * W, group_size * H * W
    group_norm_first_stage[grid](
            sample,
            output_sum,
            output_sum_squares,
            batch_stride,
            channel_stride,
            1,
            group_stride,
            num_group,
            group_size,
            # BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_G=BLOCK_SIZE_G
    )
    group_norm_second_stage[grid](
                sample,
                output,
                output_sum,
                output_sum_squares,
                weight,
                bias,
                eps,
                batch_stride,
                channel_stride,
                1,
                group_stride,
                num_group,
                group_size,
                # BLOCK_SIZE_M,
                BLOCK_SIZE_G=BLOCK_SIZE_G,
    )
    return output






if __name__ == "__main__":


    import numpy as np
    N,C,H,W,num_group = 2, 320, 128 ,128, 32
    shape_tensor_1 = paddle.to_tensor([N, C, H, W], dtype=paddle.int32)
    # paddle.seed(100)
    # sample = paddle.randn(shape_tensor_1, dtype=paddle.float32)
    sample = np.load('/nishirong/PaddleMIX/sample.npy').astype(np.float32)
    # print(sample[0][0])
    sample  = paddle.to_tensor(sample)

    # weight and bias
    weight = np.load('/nishirong/PaddleMIX/weight.npy').astype(np.float32)
    # weight = np.array(np.random.random((C)), dtype=np.float32)
    # print(weight)
    weight_tentor = paddle.to_tensor(weight)
    bias = np.load('/nishirong/PaddleMIX/bias.npy').astype(np.float32)
    # bias = np.array(np.random.random((C)), dtype=np.float32)
    # print(bias)
    bias_tentor = paddle.to_tensor(bias)
    weight_paddle = paddle.ParamAttr(name= "haha", initializer=paddle.nn.initializer.Assign(weight))
    bias_paddle = paddle.ParamAttr(name= "bias", initializer=paddle.nn.initializer.Assign(bias))
    print("=======INPUT======")
    # weight_tentor, bias_tentor, weight_paddle, bias_paddle = None, None, None, None
    
    sample_fp16 = sample.astype("float16")
    weight_tentor_fp16 = weight_tentor.astype("float16")
    bias_tensor_fp16 = bias_tentor.astype("float16")

    output = group_norm(sample.astype("float16"), num_group, 1e-5, weight_tentor.astype("float16"), bias_tentor.astype("float16")).astype("float32")

    print("====TRITON===OUTPUT=====")
    # print(sample)
    # print(weight_tentor)
    # print(bias_tentor)
    group_norm_paddle = paddle.nn.GroupNorm(num_channels=C, num_groups=num_group, weight_attr = weight_paddle ,bias_attr = bias_paddle, epsilon=1e-5)
    # print(group_norm_paddle)
    # print(weight_paddle)
    # print(bias_paddle)
    print("====PADDLE===OUTPUT=====")
    # print(sample)
    print(paddle.max(paddle.abs(sample)))
    group_norm_out = group_norm_paddle(sample)
    # print(paddle.abs(group_norm_out-output)[1][0:10])
    # print(paddle.abs(group_norm_out-output)[1][10:16])
    print(paddle.max(group_norm_out-output))
    warmup_iter = 10
    repeat_iter = 50
    # *********************
    # *****TEST TRITON*****
    # *********************
    # exit(0)
    import time
    for i in range(0, warmup_iter):
        output_test = group_norm(sample_fp16, num_group, 1e-5, weight_tentor_fp16, bias_tensor_fp16)
    paddle.device.cuda.synchronize(0)
    start = time.time()
    for i in range(0, repeat_iter):
        output_test = group_norm(sample_fp16, num_group, 1e-5, weight_tentor_fp16, bias_tensor_fp16)
    paddle.device.cuda.synchronize(0)
    end = time.time()
    print("TRITON_TIME: ", end - start)
    # *********************
    # *****TEST PADDLE*****
    # *********************
    for i in range(0, warmup_iter):
        group_norm_out_test = group_norm_paddle(sample)
    paddle.device.cuda.synchronize(0)
    start = time.time()
    for i in range(0, repeat_iter):
        group_norm_out_test = group_norm_paddle(sample)
    paddle.device.cuda.synchronize(0)
    end = time.time()
    print("PADDLE_TIME: ", end - start)
        
    




