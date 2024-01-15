


# How to build a whl for use

python3.8  -m pip wheel ./python --wheel-dir=/zhoukangkang/triton/built_wheel_by_zkk --no-deps 

# Use with PADDLE
- Set `export TRITON_USE_PADDLE=TRUE` to use TRITON with PADDLE.
