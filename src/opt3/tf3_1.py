import tensorflow as tf

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

result = a + b
print(result)

# Tensor("add:0", shape=(2,), dtype=float32)
# 节点名 第0个输出 维度 一维数组长度2 数据类型

# 计算图(Graph):搭建神经网络第计算过程，只搭建，不运算。
# y = XW = x1 * w1 + x2 * w2    乘加和
