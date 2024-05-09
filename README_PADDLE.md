# 请手动编译triton

我们想在paddle中使用triton快速实现一些算子。但因定制化、兼容性、安全等原因，可能需要对triton进行一些适应性改动。因此，选择将triton手动编译成whl的方式来安装使用，而不是直接pip install。



# 如何手动编译
- 编译环境是 gcc (GCC) 12.2.0，Ubuntu 20.04.5（示例镜像4998ed39c2b8）。
  - gcc 8.2.0，Ubuntu 18.04.6 LTS 验证为不可行。
  - 其他环境也许是可以的，但是我们没有验证过。
~~~bash
# 环境容器安装命令  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN 可选
nvidia-docker run --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --name {your-container-name} -v {hostDir}:{containerDir} --network=host -it 4998ed39c2b8 /bin/bash
# triton编译命令  wheel-dir为whl包的输出路径
python3.8 -m pip wheel ./python --wheel-dir={whlOutputPath} --no-deps 
~~~



# 在paddle中使用triton
~~~bash
# 设置如下环境变量，以在PADDLE中使用TRITON
export TRITON_USE_PADDLE=TRUE
~~~~