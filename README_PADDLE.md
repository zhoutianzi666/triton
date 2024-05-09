# 请手动编译triton

我们想在Paddle中使用triton快速实现一些算子。但因triton的部分工具类文件依赖于Torch（如，python/triton/runtime/driver.py），所以可能需要对triton进行一些Paddle兼容性改动。因此，我们需要用户 **将triton手动编译成whl并安装使用**，而 ***不是*** 直接运行`pip install triton`。


# 如何手动编译
- 编译环境: gcc (GCC) 12.2.0，Ubuntu 20.04.5。
  - gcc 8.2.0，Ubuntu 18.04.6 LTS 验证为不可行。
  - 其他环境也许是可以的，但是我们没有验证过。
- 分支选择：**main_paddle**

~~~bash
# triton编译命令:  wheel-dir为whl包的输出路径（默认为 $PWD/built_whl_triton/）
python3.8 -m pip wheel ./python --wheel-dir=built_whl_triton --no-deps 
~~~



# 在paddle中使用triton
~~~bash
# 设置如下环境变量，以在PADDLE中使用TRITON
export TRITON_USE_PADDLE=TRUE
~~~~


