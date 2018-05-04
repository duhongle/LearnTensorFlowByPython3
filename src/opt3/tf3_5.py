# 两层简单神经网络（全连接）
import tensorflow as tf

# 定义输入和参数
# 利用placeholder实现输入定义（sess.run中喂入多组数据）
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf3_5.py is:\n",
          sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]]}))
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

# 参数：即XW线上的权重W，用变量表示，随机给初值。

# 正态分布，产生2x3矩阵，标准差为2，均值为0，随机种子为1
w = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
# w = tf.truncated_normal() 去掉过大偏离点第正态分布
# w = tf.random_uniform()   平均分布

# 神经网络实现过程：
# 1.准备数据集，提取特征，作为输入喂给神经网络(Neural Network，NN)
# 2.搭建NN结构，从输入到输出(先搭建计算图，再用会话执行)（NN前向传播算法——>计算输出）
# 3.大量特征数据喂给NN，迭代优化NN参数（NN反向传播算法——>优化参数训练模型）
# 4.使用训练好的模型预测和分类


# 前向传播——>搭建模型，实现推理（以全连接网络为例）
