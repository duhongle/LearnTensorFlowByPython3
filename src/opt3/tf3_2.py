import tensorflow as tf

a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0], [4.0]])
y = tf.matmul(a, b)
print(y)

with tf.Session() as sess:
    print(sess.run(y))

# 会话(Session):执行计算图中第节点运算。
