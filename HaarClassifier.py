import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np


# 正样本加载，返回文件的路径，以list存储
positive_sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\ClassifierTrainingData\Positive\*.bmp')

# 负样本加载，返回文件的路径，以list存储
negative_sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\ClassifierTrainingData\Negative\*.jpg')

# 创建样本路径，正负样本合并
all_sample_path = positive_sample_path + negative_sample_path

# 创建样本标签
all_sample_label = []

for img in positive_sample_path:
    all_sample_label.append(1.)

for img in negative_sample_path:
    all_sample_label.append(0.)

# 对数据进行转换处理
all_sample_transform = transforms.Compose([
                transforms.Resize((96, 96)),  # 做的第一步转换，统一大小
                transforms.Grayscale(),  # 转换为灰度图像
                transforms.ToTensor()  # 第二步转换，作用：第一把PIL的Image转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])


def generate_haar_feature(image, size, haar_type):
    if haar_type == 'edge_1':
        # 生成边缘模板
        haar_template = {
            'type': [2, 1],
            'weights': [1, -1],
        }

        # 根据积分图计算矩形值
        def calc_rectangle_value(sample_integral_image, features, size, position, weights):
            feature_type_x, feature_type_y = features['type']
            feature_size = size
            feature_weight_l, feature_weight_r = weights
            x1, y1 = position
            x2, y2 = x1 + feature_size//feature_type_x, y1 + feature_size//feature_type_y
            x3, y3 = x2, y1
            x4, y4 = x3 + feature_size//feature_type_x, y2
            area_l = sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2]
            area_r = sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4]
            value = area_l * feature_weight_l + area_r * feature_weight_r
            value = value / (size * size)
            return value

        # 将数据降维成二维Tensor
        image = image.squeeze(dim=0)
        # 计算积分图
        sample_integral_image = torch.cumsum(image, dim=0).cumsum(dim=1)
        # 在进行积分图计算时需要加边进行计算
        pad_left = torch.zeros(96, 1)  # 左加边
        sample_integral_image = torch.cat([pad_left, sample_integral_image], dim=1)
        pad_top = torch.zeros(1, 97)  # 上加边
        sample_integral_image = torch.cat([pad_top, sample_integral_image], dim=0)
        # 获取该haar模板的样式
        weights = haar_template['weights']
        # 提前计算出所有位置
        positions = []
        max_steps = 96 - size + 1
        for x in range(0, max_steps, 6):
            for y in range(0, max_steps, 6):
                positions.append([x, y])
        haar_features = []
        for pos in range(len(positions)):
            haar_feature = calc_rectangle_value(sample_integral_image, haar_template, size, positions[pos], weights)
            haar_features.append(haar_feature)
    # elif haar_type == 'edge_2':
    #     # 生成线性模板
    #     haar_template = {
    #         'type': [1, 2],
    #         'weights': [1, -1],
    #     }
    # elif haar_type == 'line_1':
    #     # 生成线性模板
    #     haar_template = {
    #         'type': [3, 1],
    #         'weights': [1, -1],
    #     }
    # elif haar_type == 'line_2':
    #     # 生成线性模板
    #     haar_template = {
    #         'type': [1, 3],
    #         'weights': [1, -1],
    #     }
    # elif haar_type == 'center_1':
    #     # 生成线性模板
    #     haar_template = {
    #         'type': [2, 2],
    #         'weights': [1, -1],
    #     }
    # 转为Tensor
    haar_features = torch.tensor(haar_features)
    return haar_features


def extract_haar_features(image):
    features = []
    # 获取模板参数
    sizes = [12, 18, 24, 30, 36]
    types = ['edge_1']
    for size in sizes:
        for haar_type in types:
            # 计算haar特征
            feature = generate_haar_feature(image, size, haar_type)
            features.append(feature)
    # 拼接特征
    features = torch.cat(features)
    return features


class Mydataset(torch.utils.data.Dataset):
    # 类初始化
    def __init__(self, sample_path, sample_label, sample_transform):
        self.sample_path = sample_path
        self.sample_label = sample_label
        self.sample_transform = sample_transform

    # 进行切片
    def __getitem__(self, index):                # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        sample_path = self.sample_path[index]
        sample_label = self.sample_label[index]
        pil_img = Image.open(sample_path)
        sample_transform = self.sample_transform(pil_img)
        # Haar特征提取
        sample_features = extract_haar_features(sample_transform)
        # 返回特征和标签
        return sample_features, sample_label

    # 返回长度
    def __len__(self):
        return len(self.sample_path)


# 加载数据集
Face_dataset = Mydataset(all_sample_path, all_sample_label, all_sample_transform)

# 数据加载器
Face_dataloader = torch.utils.data.DataLoader(Face_dataset, batch_size=10, shuffle=True)


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
        # 定义可训练参数
        self.alphas = nn.Parameter(torch.ones(n_estimators))
        self.estimators = nn.ModuleList()

    def forward(self, x):
        y_pred = torch.zeros(len(x)).long()
        for alpha, estimator in zip(self.alphas, self.estimators):
            y_pred += alpha * estimator(x)
        y_pred = torch.sign(y_pred)
        return y_pred

    def fit(self, x, y):
        # 初始化学习率
        self.learning_rate = 0.1
        n_samples, n_features = x.shape
        # 初始化样本权重
        sample_weight = torch.full((n_samples,), (1 / n_samples))
        for _ in range(self.n_estimators):
            # 训练单个弱分类器
            estimator = WeakClassifier(n_features)
            estimator.fit(x, y, sample_weight)
            self.estimators.append(estimator)
            # 更新样本权重
            y_pred = estimator(x)
            sample_weight[y == y_pred] *= np.exp(-self.learning_rate)
            sample_weight /= sample_weight.sum()
            # 计算alpha
            err = (sample_weight * (y != y_pred)).sum()
            alpha = 0.5 * np.log((1 - err) / err)
            self.alphas.requires_grad_(False)
            self.alphas[_] = alpha


# 模型训练
clf = AdaBoostClassifier()
for step, (sample_features, sample_label) in enumerate(Face_dataloader):
    clf.fit(sample_features, sample_label)
    print("step:{}".format(step))

# 训练好模型后保存模型
torch.save(clf.state_dict(), r'FaceRecognition\ModelParameter\adaboost.pth')

# 加载预训练模型
clf = AdaBoostClassifier()
clf.load_state_dict(torch.load(r'FaceRecognition\ModelParameter\adaboost.pth'))
