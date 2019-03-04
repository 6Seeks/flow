
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler

# timestamp value label KPIID
data = pd.read_csv(open('C:\\Users\yangtao\Desktop\\flow\\train.csv'))
a=data.groupby(by=['KPIID'])
print(a.size())
































