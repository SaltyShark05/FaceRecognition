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
sample_num = int(len(all_sample_path))

# 创建样本标签
all_sample_label = []

for img in positive_sample_path:
    all_sample_label.append(1.)

for img in negative_sample_path:
    all_sample_label.append(-1.)

# 对数据进行转换处理
all_sample_transform = transforms.Compose([
                transforms.Resize((48, 48)),  # 做的第一步转换，统一大小
                transforms.Grayscale(),  # 转换为灰度图像
                transforms.ToTensor()  # 第二步转换，作用：第一把PIL的Image转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])


def generate_haar_feature(image, size, haar_type):
    # 根据积分图计算矩形值
    def calc_edge_1_value(sample_integral_image, features, size, position, weights):
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

    # 根据积分图计算矩形值
    def calc_line_1_value(sample_integral_image, features, size, position, weights):
        feature_type_x, feature_type_y = features['type']
        feature_size = size
        feature_weight_l, feature_weight_mid, feature_weight_r = weights
        x1, y1 = position
        x2, y2 = x1 + feature_size//feature_type_x, y1 + feature_size//feature_type_y
        x3, y3 = x2, y1
        x4, y4 = x3 + feature_size//feature_type_x, y2
        x5, y5 = x4, y1
        x6, y6 = x5 + feature_size//feature_type_x, y2
        area_l = sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2]
        area_mid = sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4]
        area_r = sample_integral_image[x5, y5] + sample_integral_image[x6, y6] - sample_integral_image[x6, y5] - sample_integral_image[x5, y6]
        value = area_l * feature_weight_l + area_mid * feature_weight_mid + area_r * feature_weight_r
        value = value / (size * size)
        return value

    # 根据积分图计算矩形值
    def calc_center_1_value(sample_integral_image, features, size, position, weights):
        feature_type_x, feature_type_y = features['type']
        feature_size = size
        feature_weight_ul, feature_weight_ur, feature_weight_dl, feature_weight_dr = weights
        x1, y1 = position
        x2, y2 = x1 + feature_size//feature_type_x, y1 + feature_size//feature_type_y
        x3, y3 = x2, y1
        x4, y4 = x3 + feature_size//feature_type_x, y2
        x5, y5 = x1, y2
        x6, y6 = x2, y2 + feature_size//feature_type_y
        x7, y7 = x2, y2
        x8, y8 = x4, y6
        area_ul = sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2]
        area_ur = sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4]
        area_dl = sample_integral_image[x5, y5] + sample_integral_image[x6, y6] - sample_integral_image[x6, y5] - sample_integral_image[x5, y6]
        area_dr = sample_integral_image[x7, y7] + sample_integral_image[x8, y8] - sample_integral_image[x8, y7] - sample_integral_image[x7, y8]
        value = area_ul * feature_weight_ul + area_ur * feature_weight_ur + area_dl * feature_weight_dl + area_dr * feature_weight_dr
        value = value / (size * size)
        return value
    # 将数据降维成二维Tensor
    image = image.squeeze(dim=0)
    # 计算积分图
    sample_integral_image = torch.cumsum(image, dim=0).cumsum(dim=1)
    # 在进行积分图计算时需要加边进行计算
    pad_left = torch.zeros(48, 1)  # 左加边
    sample_integral_image = torch.cat([pad_left, sample_integral_image], dim=1)
    pad_top = torch.zeros(1, 49)  # 上加边
    sample_integral_image = torch.cat([pad_top, sample_integral_image], dim=0)
    # 生成haar模板
    if haar_type == 'edge_1':
        # 边缘模板
        haar_template = {
            'type': [2, 1],
            'weights': [1, -1],
        }
    elif haar_type == 'line_1':
        # 线性模板
        haar_template = {
            'type': [3, 1],
            'weights': [-0.5, 1, -0.5],
        }
    elif haar_type == 'center_1':
        # 中心模板
        haar_template = {
            'type': [2, 2],
            'weights': [0.5, -0.5, -0.5, 0.5],
        }
    # 获取该haar模板的样式
    weights = haar_template['weights']
    # 提前计算出所有位置
    positions = []
    max_steps = 48 - size + 1
    for x in range(0, max_steps, 4):
        for y in range(0, max_steps, 4):
            positions.append([x, y])
    # 按照不同的模板计算不同的特征值
    if haar_type == 'edge_1':
        haar_features = []
        for pos in range(len(positions)):
            haar_feature = calc_edge_1_value(sample_integral_image, haar_template, size, positions[pos], weights)
            haar_features.append(haar_feature)
    elif haar_type == 'line_1':
        haar_features = []
        for pos in range(len(positions)):
            haar_feature = calc_line_1_value(sample_integral_image, haar_template, size, positions[pos], weights)
            haar_features.append(haar_feature)
    elif haar_type == 'center_1':
        haar_features = []
        for pos in range(len(positions)):
            haar_feature = calc_center_1_value(sample_integral_image, haar_template, size, positions[pos], weights)
            haar_features.append(haar_feature)
    # 转为Tensor
    haar_features = torch.tensor(haar_features)
    return haar_features


def extract_haar_features(image):
    features = []
    # 获取模板参数
    sizes = [6, 12, 18]
    types = ['edge_1', 'line_1', 'center_1']
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


# 划分测试集和训练集，80%数据作为训练集
index = np.random.permutation(len(all_sample_path))
all_sample_path = np.array(all_sample_path)[index]
all_sample_label = np.array(all_sample_label)[index]
divided_line = int(len(all_sample_path)*0.8)

train_imgs = all_sample_path[:divided_line]
train_labels = all_sample_label[:divided_line]
test_imgs = all_sample_path[divided_line:]
test_labels = all_sample_label[divided_line:]

# 生成数据集与数据加载器
train_dataset = Mydataset(train_imgs, train_labels, all_sample_transform)
test_dataset = Mydataset(test_imgs, test_labels, all_sample_transform)
train_batch_size = 40
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_batch_size = 200
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


# 模型定义
class WeakClassifier(nn.Module):
    def __init__(self, n_features):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = self.tanh(y_pred)
        y_pred = torch.sign(y_pred).squeeze(-1)
        return y_pred


class AdaBoostClassifier(nn.Module):
    def __init__(self, n_estimators=10):
        super(AdaBoostClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.n_features = None
        # 定义可训练参数
        self.alphas = nn.Parameter(torch.ones(n_estimators))
        self.estimators = None

    def forward(self, x):
        y_pred = torch.zeros(len(x))
        for alpha, estimator in zip(self.alphas, self.estimators):
            y_pred += alpha * estimator(x)
        y_pred = torch.sign(y_pred)
        return y_pred


# 模型训练
clf = AdaBoostClassifier()
# GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clf.to(device)
sample_features_example, sample_label_example = next(iter(train_dataloader))
if not clf.n_features:
    clf.n_features = sample_features_example.shape[1]
# 创建adaboost分类器所有包含的弱分类器
estimators = []
for _ in range(clf.n_estimators):
    estimator = WeakClassifier(n_features=clf.n_features)
    # GPU加速
    estimator.to(device)
    estimators.append(estimator)
clf.estimators = nn.ModuleList(estimators)
# 对弱分类器进行训练
for i in range(clf.n_estimators):
    # 训练单个弱分类器
    estimator = clf.estimators[i]
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(estimator.parameters(), lr=0.01)
    for step, (sample_features, sample_label) in enumerate(train_dataloader):
        # GPU加速
        sample_features = sample_features.to(device)
        sample_label = sample_label.to(device)
        optimizer.zero_grad()
        predictions = estimator(sample_features)
        sample_label = sample_label.float()
        loss = criterion(torch.sigmoid(predictions), torch.sigmoid(sample_label))
        loss.backward()
        optimizer.step()
        print('step:{}/{}'.format(step + 1, divided_line // train_batch_size))
    # 完成单个弱分类器的训练
    print("estimator[{}]/[{}] has finished training".format(i + 1, clf.n_estimators))
    clf.estimators[i] = estimator

# 对adaboost中的alpha进行训练
for step, (sample_features, sample_label) in enumerate(train_dataloader):
    # GPU加速
    sample_features = sample_features.to(device)
    sample_label = sample_label.to(device)
    # 初始化学习率
    clf.learning_rate = 0.5
    n_samples, _ = sample_features.shape
    # 初始化样本权重
    sample_weight = torch.full((n_samples,), (1 / n_samples))
    for i in range(clf.n_estimators):
        # 更新样本权重
        predictions = clf.estimators[i](sample_features)
        sample_weight[sample_label == predictions] *= np.exp(-clf.learning_rate)
        sample_weight /= sample_weight.sum()
        # 计算alpha
        err = (sample_weight * (sample_label != predictions)).sum()
        alpha = 0.5 * np.log((1 - err) / err)
        clf.alphas.requires_grad_(False)
        clf.alphas[i] = alpha
    print('step:{}/{}'.format(step + 1, divided_line // train_batch_size))

# 训练好模型后保存模型
state_dict = {
    'estimators': clf.estimators,
    'alphas': clf.alphas.detach()
}
torch.save(state_dict, r'FaceRecognition\ModelParameter\adaboost_gpu.pth')

# 加载预训练模型
clf = AdaBoostClassifier()
state_dict = torch.load(r'FaceRecognition\ModelParameter\adaboost_gpu.pth')
clf.estimators = state_dict['estimators']
clf.alphas = nn.Parameter(state_dict['alphas'])

# 设置为评估模式
clf.eval()

# 进行预测
for step, (sample_features, sample_label) in enumerate(test_dataloader):
    with torch.no_grad():  # 关闭梯度计算
        # GPU加速
        sample_features = sample_features.to(device)
        sample_label = sample_label.to(device)
        # 直接传入测试数据
        y_pred = clf(sample_features)
        # 计算指标
        acc = (y_pred == sample_label).float().mean()
        print('step:{}/{}, Accuracy:{}'.format(step + 1, (sample_num - divided_line) // test_batch_size, acc))
