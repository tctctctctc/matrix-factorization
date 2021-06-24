import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.loss_plot import semilogy
from utils.k_fold import get_k_fold_data

# 设置基础参数
batch_size = 1024
device = torch.device('cpu')
num_epochs = 50
learning_rate = 0.0006
weight_decay = 0.1


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


# 定义模型
class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bais = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)  # 0-0.05之间均匀分布
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bais.weight.data.uniform_(-0.01, 0.01)

        # 将不可训练的tensor转换成可训练的类型parameter，并绑定到module里，net.parameter()中就有了这个参数
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bais(i_id).squeeze()
        return (U * I).sum(1) + b_i + b_u + self.mean


def train(model, X_train, y_train, X_valid, y_valid, loss_func, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, valid_ls = [], []

    train_dataset = MfDataset(X_train[:, 0], X_train[:, 1], y_train)
    train_iter = DataLoader(train_dataset, batch_size)

    # 使用Adam优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = model.float()
    for epoch in range(num_epochs):
        model.train()  # 如果模型中有Batch Normalization或Dropout层，需要在训练时添加model.train()，使起作用
        total_loss, total_len = 0.0, 0
        for x_u, x_i, y in train_iter:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            l = loss_func(y_pred, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item()
            total_len += len(y)
        train_ls.append(total_loss / total_len)
        if X_valid is not None:
            model.eval()
            with torch.no_grad():
                n = y_valid.shape[0]
                valid_loss = loss_func(model(X_valid[:, 0], X_valid[:, 1]), y_valid)
            valid_ls.append(valid_loss / n)
        print('epoch %d, train mse %f, valid mse %f' % (epoch + 1, train_ls[-1], valid_ls[-1]))
    return train_ls, valid_ls


# 训练，k折交叉验证
def train_k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, num_users, num_items,
                 mean_rating):
    train_l_sum, valid_l_sum = 0.0, 0.0
    loss = torch.nn.MSELoss(reduction="sum").to(device)
    for i in range(k):
        model = MF(num_users, num_items, mean_rating).to(device)
        data = get_k_fold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(model, *data, loss, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == k:
            semilogy(range(1, num_epochs + 1), train_ls, "epochs", "mse", range(1, num_epochs + 1), valid_ls,
                     ["train", "valid"])
        print('fold %d, train mse %f, valid mse %f' % (i, train_ls[-1], valid_ls[-1]))
        print("-------------------------------------------")


def main():
    # 加载数据
    data = pd.read_csv('../dataset/u.data', header=None, delimiter='\t')
    X, y = data.iloc[:, :2], data.iloc[:, 2]
    # 转换成tensor
    X = torch.tensor(X.values, dtype=torch.int64)
    y = torch.tensor(y.values, dtype=torch.float32)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)

    mean_rating = data.iloc[:, 2].mean()
    num_users, num_items = max(data[0]) + 1, max(data[1]) + 1

    # 交叉验证选择最优超参数
    # train_k_fold(8, X_train, y_train, num_epochs=num_epochs, learning_rate=learning_rate,
    #              weight_decay=weight_decay, batch_size=batch_size, num_users=num_users, num_items=num_items,
    #              mean_rating=mean_rating)

    model = MF(num_users, num_items, mean_rating).to(device)
    loss = torch.nn.MSELoss(reduction="sum")
    train_ls, test_ls = train(model, X_train, y_train, X_test, y_test, loss, num_epochs, learning_rate, weight_decay, batch_size)

    semilogy(range(1, num_epochs + 1), train_ls, "epochs", "mse", range(1, num_epochs + 1), test_ls, ["train", "test"])
    print("\nepochs %d, mean train loss = %f, mse = %f" % (num_epochs, np.mean(train_ls), np.mean(test_ls)))

if __name__ == '__main__':
    main()
