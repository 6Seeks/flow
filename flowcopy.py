import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

data = pd.read_csv(open('C:\\Users\yangtao\Desktop\\flow\\newtrain.csv'))
value = data['value']
label = data['label']
df = pd.DataFrame()
df['A'] = value.rolling(window=100).mean()  # 移动窗口的均值
df['B'] = value.rolling(window=100).std()  # 移动窗口的标准差
df['C'] = value.rolling(window=100).median()  # 移动窗口的中位数
df['D'] = value.rolling(window=100).max()  # 移动窗口的最小值
df['E'] = value.rolling(window=100).min()  # 移动窗口的最大值
value = df.values
value = value[99:]
label = label[99:]
label = np.array(label).reshape(-1, 1)
with tf.Session() as sess:

    try:
        saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
        # 恢复所有变量信息
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
        print("成功加载模型参数")
    except:
        # 如果是第一次运行，通过init加载并初始化变量
        print("未加载模型参数，第一次运行或者模型文件被删除")
        # 获取placeholder变量
    '''
    
    value_s = tf.placeholder(tf.float32, [None, 5])
    label_s = tf.placeholder(tf.float32, [None, 1])
    prediction = sess.graph.get_tensor_by_name("prediction:0")
    print(sess.run(prediction,feed_dict={value_s: value, label_s: label}))
    '''
    graph = tf.get_default_graph()
    print_tensors_in_checkpoint_file("./checkpoint_dir/MyModel", tensor_name=None, all_tensors=False,
                                     all_tensor_names=True)
    value_s = tf.placeholder(tf.float32, [None, 5])
    label_s = tf.placeholder(tf.float32, [None, 1])

    for i in range(1000):
        print(sess.run("loss:0", feed_dict={value_s: value, label_s: label}))
