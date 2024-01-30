



# How to build a whl for use

- 编译环境是 gcc (GCC) 12.2.0，Ubuntu 20.04.5，其他环境也许是可以的，但是我们没有验证过。

python  -m pip wheel ./python --wheel-dir=/zhoukangkang/triton/built_wheel_by_zkk --no-deps 

wheel-dir 为whl包产生的文件夹路径

# Use with PADDLE
- Set `export TRITON_USE_PADDLE=TRUE` to use TRITON with PADDLE.
