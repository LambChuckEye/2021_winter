import numpy as np
import pandas as pd
from pandas import DataFrame
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 读取文件
raw_data = pd.read_csv('data/train.csv')

# 1、获取数字类型的数据：
numeric_colmuns = []
#   pd.dtypes               查看所有属性类型
#   pd.dtypes['属性名']      查看指定属性类型
#   pd.dtypes.index         返回属性名列表
print(list(raw_data.dtypes.index))
numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.int64].index))
numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == np.float64].index))
# 将价格置于最后一位
numeric_colmuns.remove('SalePrice')
numeric_colmuns.append('SalePrice')
# 去除id属性（非特征）
numeric_colmuns.remove('Id')
# 获取属性数据
numeric_data = DataFrame(raw_data, columns=numeric_colmuns)

# 处理Nan数据
#   pd.isna(DATA) 将所有Nan设为True
#   np.any(A==B)    存在True则为True
#   np.all(A==B)    全部位True则为True
# 获取存在Nan的字段
nan_columns = np.any(pd.isna(numeric_data), axis=0)
nan_columns = list(nan_columns[nan_columns == True].index)
# 将nan变为0
for i in nan_columns:
    numeric_data[i] = numeric_data[i].fillna(0)

# 2、模型搭建
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 制作x,y
numeric_x_columns = list(numeric_data.columns)
numeric_x_columns.remove('SalePrice')
numeric_y_columns = ['SalePrice']

# 标准化
numeric_data = (numeric_data - numeric_data.mean()) / numeric_data.std()

numeric_x_df = DataFrame(numeric_data, columns=numeric_x_columns)
numeric_y_df = DataFrame(numeric_data, columns=numeric_y_columns)
# 转换tensor
numeric_x = torch.tensor(numeric_x_df.values, dtype=torch.float)
numeric_y = torch.tensor(numeric_y_df.values, dtype=torch.float)


class Net(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1).cuda()
        self.linear2 = nn.Linear(H1, H2).cuda()
        self.linear3 = nn.Linear(H2, H3).cuda()
        self.linear4 = nn.Linear(H3, D_out).cuda()

    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred


H1, H2, H3 = 500, 1000, 200
D_in, D_out = numeric_x.shape[1], numeric_y.shape[1]
model1 = Net(D_in, H1, H2, H3, D_out)
# 均方误差
criterion = nn.MSELoss()
# 随机梯度下降
optimizer = torch.optim.SGD(model1.parameters(), lr=1e-4 * 2)
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4 * 2)

losses = []
device = d2l.try_gpu()
# gpu训练
model1.to(device)
# tensor添加gpu需要赋值
numeric_y = numeric_y.to(device)
numeric_x = numeric_x.to(device)

# for t in range(500):
#     # 向前传播
#     y_pred = model1(numeric_x)
#     # 计算损失
#     loss = criterion(y_pred, numeric_y)
#     print(t, loss.item())
#     losses.append(loss.item())
#
#     if torch.isnan(loss):
#         break
#     # 梯度清零
#     optimizer.zero_grad()
#     # 向后传播
#     loss.backward()
#     # 赋值
#     optimizer.step()

plt.figure(figsize=(12, 10))
plt.plot(range(len(losses)), losses)
plt.show()


# ============== 文字
# 文字字段
non_numeric_columns = [col for col in list(raw_data.columns) if col not in numeric_colmuns]
non_numeric_columns.remove('Id')

non_numeric_data = DataFrame(raw_data, columns=non_numeric_columns)
# 处理Nan
nan_columns = np.any(pd.isna(non_numeric_data), axis=0)
nan_columns = list(nan_columns[nan_columns == True].index)
for col in nan_columns:
    non_numeric_data[col] = non_numeric_data[col].fillna('N/A')

# 字段映射
mapping_table = dict()

for col in non_numeric_columns:
    curr_mapping_table = dict()

    unique_values = pd.unique(non_numeric_data[col])
    for inx, v in enumerate(unique_values):
        curr_mapping_table[v] = inx + 1
        non_numeric_data[col] = non_numeric_data[col].replace(v, inx + 1)

    mapping_table[col] = curr_mapping_table

# 归一化
non_numeric_data = (non_numeric_data - non_numeric_data.mean()) / (non_numeric_data.max() - non_numeric_data.min())

non_numeric_x_df = DataFrame(non_numeric_data, columns=non_numeric_columns)
non_numeric_y_df = DataFrame(numeric_y_df)
non_numeric_x = torch.tensor(non_numeric_x_df.values, dtype=torch.float)
non_numeric_y = torch.tensor(non_numeric_y_df.values, dtype=torch.float)

D_in, D_out = non_numeric_x.shape[1], non_numeric_y.shape[1]

model5 = Net(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.Adam(model5.parameters(), lr=1e-4 * 2)

device = d2l.try_gpu()
# gpu训练
model5.to(device)
# tensor添加gpu需要赋值
non_numeric_y = non_numeric_y.to(device)
non_numeric_x = non_numeric_x.to(device)

losses5 = []

# for t in range(500):
#     y_pred = model5(non_numeric_x)
#
#     loss = criterion(y_pred, non_numeric_y)
#     print(t, loss.item())
#     losses5.append(loss.item())
#
#     if torch.isnan(loss):
#         break
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

plt.figure(figsize=(12, 10))
plt.plot(range(len(losses5)), losses5, label = 'Non-Numeric')

plt.legend(loc='upper right')
plt.show()


# 所有数据

x_df = DataFrame(numeric_x_df, columns=numeric_x_columns)
y_df = DataFrame(numeric_y_df)
for col in non_numeric_columns:
    x_df[col] = non_numeric_x_df[col]

x = torch.tensor(x_df.values, dtype=torch.float)
y = torch.tensor(y_df.values, dtype=torch.float)

D_in, D_out = x.shape[1], y.shape[1]
model6 = Net(D_in, H1, H2, H3, D_out)
optimizer = torch.optim.Adam(model6.parameters(), lr=1e-4 * 2)
losses6 = []

device = d2l.try_gpu()
# gpu训练
model6.to(device)
# tensor添加gpu需要赋值
y = y.to(device)
x = x.to(device)

for t in range(500):
    y_pred = model6(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())
    losses6.append(loss.item())

    if torch.isnan(loss):
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.figure(figsize=(12, 10))
plt.plot(range(len(losses6)), losses6, label = 'Entire Data')

plt.legend(loc='upper right')
plt.show()