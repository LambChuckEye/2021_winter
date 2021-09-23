import numpy as np
import pandas as pd
from pandas import DataFrame
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# 获取数组和非数字字段
def get_numeric_and_non(raw_data):
    # 数字字段
    numeric_colmuns = []
    # int和float
    numeric_colmuns.extend(list(raw_data.dtypes[raw_data.dtypes == (np.int64 or np.float64)].index))
    # numeric_colmuns.remove('SalePrice')
    # numeric_colmuns.append('SalePrice')
    # 文字字段
    non_numeric_columns = [col for col in list(raw_data.columns) if col not in numeric_colmuns]
    numeric_colmuns.remove('Id')

    numeric_data = DataFrame(raw_data, columns=numeric_colmuns)
    non_numeric_data = DataFrame(raw_data, columns=non_numeric_columns)

    return numeric_data, non_numeric_data


def deal_nan(data, fill):
    nan_columns = np.any(pd.isna(data), axis=0)
    # 获取标签
    nan_columns = list(nan_columns[nan_columns == True].index)
    # 将nan变为0
    for i in nan_columns:
        data[i] = data[i].fillna(fill)
    return data


def mapping(data):
    mapping_table = dict()
    for col in list(data.dtypes.index):
        curr_mapping_table = dict()
        unique_values = pd.unique(data[col])
        for inx, v in enumerate(unique_values):
            curr_mapping_table[v] = inx + 1
            data[col] = data[col].replace(v, inx + 1)
        mapping_table[col] = curr_mapping_table
    return mapping_table


def norm(data):
    means, maxs, mins = dict(), dict(), dict()
    for col in data:
        means[col] = data[col].mean()
        maxs[col] = data[col].max()
        mins[col] = data[col].min()
    return (data - data.mean()) / (data.max() - data.min()), means, maxs, mins


def load(filename):
    raw_data = pd.read_csv(filename)
    # 获取文字与数字部分
    numeric_data, non_numeric_data = get_numeric_and_non(raw_data)
    # 去除nan
    numeric_data = deal_nan(numeric_data, 0)
    non_numeric_data = deal_nan(non_numeric_data, 'N/A')
    # 文字部分做映射
    mapping_table = mapping(non_numeric_data)
    # 合并
    return pd.concat([non_numeric_data, numeric_data], axis=1)


def train(model, x, y, loss, optimizer, epoch):
    # gpu训练
    device = d2l.try_gpu()
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    losses = []

    for t in range(epoch):
        y_pred = model(x)
        l = loss(y_pred, y)
        print(t, l.item())
        losses.append(l.item())
        if torch.isnan(l):
            break
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    return model, losses


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def prediction(data, model, device):
    data = data.to(device)
    return model(data)


if __name__ == '__main__':
    train_data = load('data/train.csv')
    features_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    train_data.drop(features_to_drop, axis=1, inplace=True)
    # 标准化
    train_data, means, maxs, mins = norm(train_data)
    # 分离x，y
    y = train_data[['SalePrice']]
    x = train_data.drop(['SalePrice'], axis=1)
    # 生成tensor向量
    x = torch.tensor(x.values, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)

    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
                        nn.Linear(512, 1))
    # 模型构建
    net.apply(init_weights)
    # 均方误差
    loss = nn.MSELoss()  # Adam优化，不依赖与学习率的选择
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4 * 5, weight_decay=0.001)
    # 训练
    model, losses = train(net, x, y, loss, optimizer, 1000)

    # 画图
    plt.figure(figsize=(12, 10))
    plt.plot(range(len(losses)), losses)
    plt.show()

    # 测试集预测
    test_data = load('data/test.csv')
    test_data.drop(features_to_drop, axis=1, inplace=True)
    test_data, _, _, _ = norm(test_data)

    test_tensor = torch.tensor(test_data.values, dtype=torch.float)

    y_hat = prediction(test_tensor, model, d2l.try_gpu())
    # y_hat = prediction(x, model, d2l.try_gpu())

    y_hat = y_hat * (maxs['SalePrice'] - mins['SalePrice']) + means['SalePrice']

    raw_data = pd.read_csv('data/test.csv')
    # raw_data = pd.read_csv('data/train.csv')

    result = raw_data[['Id']]
    result['SalePrice'] = y_hat.cpu().detach().numpy()
    print(result.head(10))
    result.to_csv('result3.csv', index=None)
