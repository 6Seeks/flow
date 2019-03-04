import tensorflow as tf
import numpy as np
import pandas as pd
# 引入imblearn的原因是因为，成功的label太多了，是不成功的20倍，如果不平衡这种，成功的权重太大，在神经网络中，就会一直认为正确，导致的结果就是：
# 除了开始的几次学习可以计算出结果，后来的结果都是nan
from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.ensemble import EasyEnsemble  # 简单集成方法EasyEnsemble
from imblearn.combine import SMOTEENN


# 添加层
# 添加层不变
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 1.训练的数据
# Make up some real data5o'k'kx_data = np.linspace(-1,1,300)[:, np.newaxis]
# timestamp value label KPIID 真正有用的就是 value label，时间间隔相同
data = pd.read_csv(open('C:\\Users\yangtao\Desktop\\flow\\newtrain.csv'))
x = np.array(data['value']).reshape(-1, 1)  # 需要转置
y = np.array(data['label']).reshape(-1, 1)

# 欠抽样处理
'''
# 使用RandomUnderSampler方法进行欠抽样处理

model_RandomUnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(x,
                                                                                                     y)  # 输入数据并作欠抽样处理
'''

smote_enn = SMOTEENN(random_state=0)
#x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = smote_enn.fit_resample(x, y)#训练的时候用这个，另一个注释掉
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =(x, y)#用于测试的,测试的时候用这个，另一个注释掉
# label=np.array(data['label']).reshape([-1,1])#转成tensor 1维的
#####预处理
x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled)  # 由np.array再转回来dataframe
df = pd.DataFrame()
# x_RandomUnderSampler_resampled预处理
t_mean = pd.DataFrame(x_RandomUnderSampler_resampled.rolling(window=100).mean())
t_std = pd.DataFrame(x_RandomUnderSampler_resampled.rolling(window=100).std())
t_median = pd.DataFrame(x_RandomUnderSampler_resampled.rolling(window=100).median())
t_max = pd.DataFrame(x_RandomUnderSampler_resampled.rolling(window=100).max())
t_min = pd.DataFrame(x_RandomUnderSampler_resampled.rolling(window=100).min())
# list原因 ValueError: Cannot set a frame with no defined index and a value that cannot be converted to a Series
df['A'] = list(t_mean.values)  # 移动窗口的均值
df['B'] = list(t_std.values)  # 移动窗口的标准差
df['C'] = list(t_median.values)  # 移动窗口的中位数
df['D'] = list(t_max.values)  # 移动窗口的最小值
df['E'] = list(t_min.values)  # 移动窗口的最大值

value = df.values
# 数据切片,把前面的nan去掉
value = value[99:]
label = y_RandomUnderSampler_resampled[99:]
label = np.array(label).reshape(-1, 1)
# 2.定义节点准备接收数据
# 可以这样认为，value是自变量，而label是函数值
# define placeholder for inputs to network
value_s = tf.placeholder(tf.float32, [None, 5])
label_s = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# 我们一次取10个数据，5X10，10X1
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
l1 = add_layer(value_s, 5, 10, activation_function=tf.sigmoid)


# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1,10, 1, activation_function=None)
# 4.定义 loss 表达式
# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(label_s - prediction), reduction_indices=[1]), name='loss')

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step 对所有变量进行初始化
saver = tf.train.Saver()  # 存储训练模型
#  init = tf.initialize_all_variables()

init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
try:
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    # 恢复所有变量信息
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    print("成功加载模型参数")
    a = sess.run(prediction, feed_dict={value_s: value, label_s: label})
    print(len(a), len(label))
    right = 0
    sum = 0
    for i in range(len(a)):
        if label[i] == 1:
            sum = sum + 1

            if a[i] > 0.5:
                right = right + 1
        if label[i] == 0:
            sum = sum + 1

            if a[i] < 0.5:
                right = right + 1

    print(right / sum * 100, "%")
    right = 0
    sum = 0
    for i in range(len(a)):
        if label[i]==0 and  a[i] > 0.5:
            right=right+1
    print(right/len(a)*100,'%')







except:
    # 如果是第一次运行，通过init加载并初始化变量
    print("未加载模型参数，第一次运行或者模型文件被删除")
    # 获取placeholder变量

    sess.run(init)

    # 迭代 1000 次学习，sess.run optimizer

    for i in range(1000):
        # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数

        sess.run(train_step, feed_dict={value_s: value, label_s: label})

        if i % 5 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={value_s: value, label_s: label}))

    saver.save(sess, './checkpoint_dir/MyModel')
