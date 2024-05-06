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
    N,
    C,
    H,
    W,
    num_groups,
    BLOCK_SIZE_G: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channels_num = C
    group_size = C // num_groups
    channels_stride = H * W
    group_stride = group_size * channels_stride
    batch_stride = C * H * W
    group_bias = batch_id * batch_stride + group_id * group_stride
    _zeros = 0.0
    _sum = 0.0
    _sum_squares = 0.0
    # 求和
    offset_block = tl.arange(0, BLOCK_SIZE * BLOCK_SIZE_G)
    sample_ptrs = sample_ptr + group_bias + offset_block
    
    for k in range(0, tl.cdiv(group_stride, BLOCK_SIZE * BLOCK_SIZE_G)):
        sample_ = tl.load(sample_ptrs, mask= (offset_block < group_stride - k * BLOCK_SIZE * BLOCK_SIZE_G), other=0.0)
        sample = sample_.to(tl.float32)
        _sum = _sum + tl.sum(sample)
        _sum_squares = _sum_squares + tl.sum(sample * sample)
        sample_ptrs += BLOCK_SIZE * BLOCK_SIZE_G
    
    # offset_channel = (tl.arange(0, BLOCK_SIZE_G))
    # offset_group = tl.arange(0, BLOCK_SIZE) 

    # for k in range(0, tl.cdiv(channels_stride, BLOCK_SIZE)):
    #     cc_mask = ((offset_group[None, :] < channels_stride - k * BLOCK_SIZE) & (channel_mask))
    #     sample_ = tl.load(sample_ptrs, mask= cc_mask, other=_zeros)
    #     sample = sample_.to(tl.float32)
    #     _sum = _sum + tl.sum(sample)
    #     _sum_squares = _sum_squares + tl.sum(sample * sample)
    #     sample_ptrs += BLOCK_SIZE
    
    _mean = _sum / group_stride
    _var = _sum_squares / group_stride - _mean * _mean
    
    # 求归一化
    rstd = 1 / tl.sqrt(_var + eps)
    offset_channel = (tl.arange(0, BLOCK_SIZE_G)) % group_size
    offset_group = tl.arange(0, BLOCK_SIZE) 
    ptrs_offset = group_bias + offset_channel[:, None] * channels_stride + offset_group[None,:]
    sample_ptrs = sample_ptr + ptrs_offset
    output_ptrs = output_ptr + ptrs_offset
    weight_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    bias_para = tl.zeros((BLOCK_SIZE_G, 1), dtype= tl.float32)
    if weight_ptr: 
        weight_para_temp = tl.load(weight_ptr + group_id * group_size + offset_channel[:, None])
        weight_para = weight_para_temp.to(tl.float32)
    if bias_ptr:
        bias_para_temp = tl.load(bias_ptr + group_id * group_size + offset_channel[:, None])
        bias_para = bias_para_temp.to(tl.float32)
    
    for k in range(0, tl.cdiv(channels_stride, BLOCK_SIZE)):
        cc_mask = ((offset_group[None, :] < channels_stride - k * BLOCK_SIZE))

        sample_ = tl.load(sample_ptrs, mask = cc_mask, other = _zeros)
        sample = sample_.to(tl.float32)
        re = (sample - _mean) * rstd
        if weight_ptr:
            re = re * weight_para
        if bias_ptr:
            re = re + bias_para
        re = re.to(tl.float16)
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
    # print(sample.dtype)
    output = paddle.empty((N, C, H, W), dtype=sample.dtype)
    group_norm_kernel[grid](sample, 
                            output,
                            weight,
                            bias,
                            eps,
                            N,
                            C,
                            H,
                            W,
                            num_group,
                            BLOCK_SIZE=128,
                            BLOCK_SIZE_G = triton.next_power_of_2(group_size),
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
        
    




