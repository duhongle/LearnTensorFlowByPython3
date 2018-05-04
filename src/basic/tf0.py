import tensorflow as tf
import numpy as np

# 生成0和1矩阵
v1 = tf.Variable(tf.zeros([4, 3, 2]), name="v1")
v2 = tf.Variable(tf.ones([6, 4, 3]), name="v2")

# 填充单值矩阵
v3 = tf.Variable(tf.fill([2, 3], 9))

# 常量矩阵
v4_1 = tf.constant([1, 2, 3, 4, 5, 6, 7])
v4_2 = tf.constant(-1, shape=[2, 3])

# 生成等差数列
v6_1 = tf.linspace(10.0, 14.0, 6, name='linspace')
v7_1 = tf.range(10, 20, 2)

# 生成各种随机数据矩阵
v8_1 = (tf.random_uniform([2, 4], minval=0.0, maxval=2.0, dtype=tf.float32, seed=1234, name="v8_1"))
v8_2 = (tf.random_normal([2, 3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1234, name="v8_2"))
v8_3 = (tf.truncated_normal([2, 3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1234, name='v8_3'))
v8_4 = (tf.random_uniform([2, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1234, name='v8_4'))
v8_5 = (tf.random_shuffle([[1, 2, 3], [4, 5, 6], [6, 6, 6]], seed=134, name='v8_5'))

# 保存变量，也可以指定保存的内容
saver = tf.train.Saver()

# 运行
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # numpy保存文件
    np.save("v1.npy", sess.run(v1))
    test_a = np.load("v1.npy")
    print(test_a[1, 2])

    v5 = tf.zeros_like(v3)
    print(v5)

    for vi in [v1, v2, v3, v4_1, v4_2, v5, v6_1, v7_1, v8_1, v8_2, v8_3, v8_4, v8_5]:
        print("类型:%s\n形状:%s\n值:%s\n" % (vi, tf.Variable.get_shape(vi), sess.run(vi)))

    # 保存图的变量
    save_path = saver.save(sess, "../tmp/model.ckpt")
    # 加载图的变量
    # saver.restore(sess, "../tmp/model.ckpt")

    print("Model saved in file:", save_path)
