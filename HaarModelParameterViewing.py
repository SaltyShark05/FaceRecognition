import torch
import torch.nn as nn
import numpy as np


# 模型定义
class WeakClassifier(nn.Module):
    def __init__(self, n_features):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        y_pred = torch.round(y_pred).squeeze(-1)
        return y_pred

    def fit(self, x, y, sample_weight):
        criterion = nn.BCELoss(reduction='none')
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for _ in range(100):
            y_pred = self(x)
            y = y.float()
            loss = criterion(y_pred, y)
            loss = (loss * sample_weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class AdaBoostClassifier(nn.Module):
    def __init__(self, n_estimators=100):
        super(AdaBoostClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.n_features = None
        # 定义可训练参数
        self.alphas = nn.Parameter(torch.ones(n_estimators))
        self.estimators = None

    def forward(self, x):
        y_pred = torch.zeros(len(x)).long()
        for alpha, estimator in zip(self.alphas, self.estimators):
            y_pred += torch.tensor(alpha * estimator(x)).long()
        y_pred = torch.sign(y_pred)
        return y_pred

    def fit(self, x, y):
        if not self.n_features:
            self.n_features = x.shape[1]
        if not self.estimators:
            estimators = [WeakClassifier(n_features=self.n_features) for _ in range(self.n_estimators)]
            self.estimators = nn.ModuleList(estimators)
        # 初始化学习率
        self.learning_rate = 0.1
        n_samples, _ = x.shape
        # 初始化样本权重
        sample_weight = torch.full((n_samples,), (1 / n_samples))
        for i in range(self.n_estimators):
            # 训练单个弱分类器
            estimator = self.estimators[i]
            estimator.fit(x, y, sample_weight)
            self.estimators[i] = estimator
            # 更新样本权重
            y_pred = self.estimators[i](x)
            sample_weight[y == y_pred] *= np.exp(-self.learning_rate)
            sample_weight /= sample_weight.sum()
            # 计算alpha
            err = (sample_weight * (y != y_pred)).sum()
            alpha = 0.5 * np.log((1 - err) / err)
            self.alphas.requires_grad_(False)
            self.alphas[i] = alpha


clf = AdaBoostClassifier()
state_dict = torch.load(r'FaceRecognition\ModelParameter\adaboost.pth')
print(state_dict)
