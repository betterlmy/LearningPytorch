import hashlib, os, tarfile, zipfile, requests

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import d2l
import lmy
from icecream import ic

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('.', 'data')):
    """下载DATAHUB中的文件，并返回本地文件名"""
    assert name in DATA_HUB, f"{name}不存在于DATA_HUB"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            print("本地已经存在")
            return fname  # 本地命中
    print(f"正在从{url}下载{fname}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print("下载完成")
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, "只有zip或者tar，gz文件可以被解压"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


def data_prepare():
    """
    下载数据 并进行预处理
    :return:
    """

    # 数据下载
    DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce'
    )
    DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
    )
    train_data = pd.read_csv(download('kaggle_house_train'))
    # lmy.print_shape(train_data) #1460 x (80 + 1) 1460个数据 80个特征  一个标签（房价)
    test_data = pd.read_csv(download('kaggle_house_test'))
    # lmy.print_shape(test_data) # 1459 x (80 + 1)

    # 数据预处理
    # 查看部分数据的特征和标签
    # print(train_data.iloc[:4, [0, 1, 2, 3, -2, -1]])
    train_features = train_data.iloc[:, 1:-1]
    train_labels = train_data.iloc[:, -1]
    # print(train_features.iloc[:4, [0, 1, 2, 3, -2, -1]])
    # print(train_labels[:4])
    test_features = test_data.iloc[:, 1:]
    # print(test_features.iloc[:4, [0, 1, 2, 3, -2, -1]])
    all_features = pd.concat((train_features, test_features))
    del train_features
    del test_features
    # lmy.print_shape(all_features)

    # 数据标准化 过程
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 这些特征是数字类型的特征
    lmy.print_shape(all_features[numeric_features].iloc[:4, :3])
    # 归一化
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # Nan填充
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # lmy.print_shape(all_features[numeric_features].iloc[:4, :3])
    all_features = pd.get_dummies(all_features, dummy_na=True)  # dummy_na将缺失值视为有效的特征值，并为其创建指示符
    # all_features.shape  # 可以看到特征的数量已经从79个增加到331个
    num_train = train_data.shape[0]

    # 通过values转为Numpy，再通过torch.tensor转为tensor属性
    train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_labels.values, dtype=torch.float32)
    return train_features, test_features, train_labels


def get_net(num_features):
    return nn.Sequential(nn.Linear(num_features, 1))


def log_rmse(net, features, labels, loss):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(labels), torch.log(clipped_preds)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, loss, num_epochs, lr, weight_decay,
          batch_size):
    train_list, test_list = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()  # 初始化
            l = loss(net(X), y)
            l.backward()
            optimizer.step()

        train_list.append(log_rmse(net, train_features, train_labels,loss))
        if test_labels is not None:
            test_list.append(log_rmse(net, test_features, test_labels,loss))
    return train_list, test_list


def get_k_fold_data(k, i, X, y):
    assert k > 1, "k<1"
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        print(idx)
        # part表示当前进行验证的数据
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, loss, num_epochs, num_features, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(num_features)
        train_ls, valid_ls = train(net, *data, loss, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse',
                     xlim=[1, num_epochs], legend=['train', 'valid'], yscale='log')
        print(f"折{i + 1},训练log rmse{float(train_ls[-1]):.5f},"
              f"验证log rmse{float(valid_ls[-1]):.5f}")
    return train_l_sum / k, valid_l_sum / k


def main():
    train_features, test_features, train_labels = data_prepare()
    ic(train_labels.shape)
    ic(train_features.shape)
    ic(test_features.shape)
    loss = nn.MSELoss()
    num_features = train_features.shape[1]
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_ls, valid_ls = k_fold(k, train_features, train_labels, loss, num_epochs, num_features, lr, weight_decay,
                                batch_size)


if __name__ == '__main__':
    main()
