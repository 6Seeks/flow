import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler

# timestamp value label KPIID
data = pd.read_csv(open('C:\\Users\yangtao\Desktop\\flow\\newtrain.csv'))
time=data['timestamp']
value=data['value']
label=data['label']
plt.hist(x=value,bins=np.arange(0,8,0.05))
plt.show()
