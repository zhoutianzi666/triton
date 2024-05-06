1. group-norm-v0.py ，naive版本，只用一个triton kernel完成计算，方差计算也是最原始的。
2. group-norm-v1.py ，将方差计算用x2求和优化。
3. group-norm-v2.py ，split版本，拆分成两个算子来实现，第一个算子计算组内的和以及平方和，第二个算子计算最后的结果。
