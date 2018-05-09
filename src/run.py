import tensorflow as tf
import numpy as np
import os

print(tf.__version__, tf.__path__)

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 启动图 (graph)

# 控制使用哪块GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使用GPU 0,1

# 控制GPU资源使用率
# allow growth
# log_device_placement=True 是否打印设备分配信息
# allow_soft_placement=True 如果你指定的设备不存在，运行TF自动分配设备
# 设置每个GPU应该拿出多少容量给进程使用，0.7代表70%
config = tf.ConfigProto(log_device_placement=False,
                        allow_soft_placement=True,
                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
# 使用allow_growth option,刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
config.gpu_options.allow_growth = True

queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.int64])
# enqueue_many的写法，两个元素放在两个列表里。
en_m = queue.enqueue_many([["hello", "world"], [1, 2]])

# enqueue的写法
en = queue.enqueue(["hello", 1])
deq = queue.dequeue()

with tf.Session(config=config) as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(en_m))
    print(sess.run(en))
    print(sess.run(deq))

    # 拟合平面
    for step in range(0, 1001):
        sess.run(train)
        if step % 100 == 0:
            print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
