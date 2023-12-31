import glob
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np


# 正样本加载，返回文件的路径，以list存储
positive_sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\ClassifierTrainingData\Positive\*.jpg')

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
    def calc_edge_1_value(sample_integral_image, features, size, positions, weights):
        x1, y1 = positions[:, 0], positions[:, 1]
        x2, y2 = x1 + size // features['type'][0], y1 + size // features['type'][1]
        x3, y3 = x2, y1
        x4, y4 = x3 + size//features['type'][0], y2
        area_l = (sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2])
        area_r = (sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4])
        value = (area_l * weights[0] + area_r * weights[1]) / (size * size)
        return value

    def calc_line_1_value(sample_integral_image, features, size, positions, weights):
        x1, y1 = positions[:, 0], positions[:, 1]
        x2, y2 = x1 + size // features['type'][0], y1 + size // features['type'][1]
        x3, y3 = x2, y1
        x4, y4 = x3 + size // features['type'][0], y2
        x5, y5 = x4, y1
        x6, y6 = x5 + size // features['type'][0], y2
        area_l = (sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2])
        area_mid = (sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4])
        area_r = (sample_integral_image[x5, y5] + sample_integral_image[x6, y6] - sample_integral_image[x6, y5] - sample_integral_image[x5, y6])
        value = (area_l * weights[0] + area_mid * weights[1] + area_r * weights[2]) / (size * size)
        return value

    def calc_center_1_value(sample_integral_image, features, size, positions, weights):
        x1, y1 = positions[:, 0], positions[:, 1]
        x2, y2 = x1 + size // features['type'][0], y1 + size // features['type'][1]
        x3, y3 = x2, y1
        x4, y4 = x3 + size // features['type'][0], y2
        x5, y5 = x1, y2
        x6, y6 = x2, y2 + size // features['type'][1]
        x7, y7 = x2, y2
        x8, y8 = x4, y6
        area_ul = (sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2])
        area_ur = (sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4])
        area_dl = (sample_integral_image[x5, y5] + sample_integral_image[x6, y6] - sample_integral_image[x6, y5] - sample_integral_image[x5, y6])
        area_dr = (sample_integral_image[x7, y7] + sample_integral_image[x8, y8] - sample_integral_image[x8, y7] - sample_integral_image[x7, y8])
        value = (area_ul * weights[0] + area_ur * weights[1] + area_dl * weights[2] + area_dr * weights[3]) / (size * size)
        return value

    # 将数据降维成二维Tensor
    image = image.squeeze(dim=0)
    # 计算积分图
    sample_integral_image = torch.cumsum(image, dim=0, dtype=torch.float32).cumsum(dim=1, dtype=torch.float32)
    # 在进行积分图计算时需要加边进行计算
    sample_integral_image = torch.nn.functional.pad(sample_integral_image, (1, 0, 1, 0))
    # 生成haar模板
    if haar_type == 'edge_1':
        haar_template = {'type': [2, 1], 'weights': [1, -1]}
    elif haar_type == 'line_1':
        haar_template = {'type': [3, 1], 'weights': [-0.5, 1, -0.5]}
    elif haar_type == 'center_1':
        haar_template = {'type': [2, 2], 'weights': [0.5, -0.5, -0.5, 0.5]}

    weights = haar_template['weights']
    positions = torch.tensor([[x, y] for x in range(0, 48 - size + 1, 4) for y in range(0, 48 - size + 1, 4)])

    if haar_type == 'edge_1':
        haar_features = calc_edge_1_value(sample_integral_image, haar_template, size, positions, weights)
    elif haar_type == 'line_1':
        haar_features = calc_line_1_value(sample_integral_image, haar_template, size, positions, weights)
    elif haar_type == 'center_1':
        haar_features = calc_center_1_value(sample_integral_image, haar_template, size, positions, weights)

    return torch.tensor(haar_features)


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

divided_line_1 = int(len(all_sample_path)*0.2)
divided_line_2 = int(len(all_sample_path)*0.8)

weak_train_imgs = all_sample_path[:divided_line_1]
weak_train_labels = all_sample_label[:divided_line_1]
ada_train_imgs = all_sample_path[divided_line_1:divided_line_2]
ada_train_labels = all_sample_label[divided_line_1:divided_line_2]
test_imgs = all_sample_path[divided_line_2:]
test_labels = all_sample_label[divided_line_2:]

# 生成数据集与数据加载器
weak_train_dataset = Mydataset(weak_train_imgs, weak_train_labels, all_sample_transform)
ada_train_dataset = Mydataset(ada_train_imgs, ada_train_labels, all_sample_transform)
test_dataset = Mydataset(test_imgs, test_labels, all_sample_transform)
weak_train_batch_size = 200
weak_train_dataloader = torch.utils.data.DataLoader(weak_train_dataset, batch_size=weak_train_batch_size, shuffle=True)
ada_train_batch_size = 200
ada_train_dataloader = torch.utils.data.DataLoader(ada_train_dataset, batch_size=ada_train_batch_size, shuffle=True)
test_batch_size = 800
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


# 模型定义
class WeakClassifier(nn.Module):
    def __init__(self, n_features, n_hidden=32, n_output=1):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = y_pred.squeeze(-1)
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
        # GPU加速
        y_pred = y_pred.to(device)
        for alpha, estimator in zip(self.alphas, self.estimators):
            # GPU加速
            alpha = alpha.to(device)
            estimator = estimator.to(device)
            y_pred += alpha * estimator(x)
        y_pred = torch.sign(y_pred)
        return y_pred

# GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型训练
clf = AdaBoostClassifier()
clf.to(device)
# 确定样本形状
sample_features_example, sample_label_example = next(iter(weak_train_dataloader))
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
    optimizer = torch.optim.Adam(estimator.parameters(), lr=0.05)
    epochs = 10
    for epoch in range(epochs):
        for step, (sample_features, sample_label) in enumerate(weak_train_dataloader):
            # GPU加速
            sample_features = sample_features.to(device)
            sample_label = sample_label.to(device)
            optimizer.zero_grad()
            predictions = estimator(sample_features)
            sample_label = sample_label.float()
            loss = criterion(torch.sigmoid(predictions), torch.sigmoid(sample_label))
            loss.backward()
            optimizer.step()
            print('estimator[{}]/[{}], epoch:{}/{}, step:{}/{}'.format(i + 1, clf.n_estimators, epoch + 1, epochs, step + 1, divided_line_1 // weak_train_batch_size))
    # 进行一次检验
    sample_features_example, sample_label_example = next(iter(weak_train_dataloader))
    # GPU加速
    sample_features_example = sample_features.to(device)
    sample_label_example = sample_label.to(device)
    y_pred = torch.sign(estimator(sample_features_example))
    # 计算指标
    acc = (y_pred == sample_label_example).float().mean()
    print('estimator[{}]/[{}], Accuracy:{}'.format(i + 1, clf.n_estimators, acc))
    clf.estimators[i] = estimator

# 对adaboost中的alpha进行训练
for step, (sample_features, sample_label) in enumerate(ada_train_dataloader):
    # GPU加速
    sample_features = sample_features.to(device)
    sample_label = sample_label.to(device)
    n_samples, _ = sample_features.shape
    # GPU加速
    for i in range(clf.n_estimators):
        # 更新样本权重
        predictions = torch.sign(clf.estimators[i](sample_features))
        # 计算alpha
        err = (sample_label != predictions).float().mean()
        if err == 0:
            err = 0.001
        alpha = 0.5 * np.log((1 - err.cpu().numpy()) / err.cpu().numpy())
        clf.alphas.requires_grad_(False)
        clf.alphas[i] = alpha
    print('step:{}/{}'.format(step + 1, (divided_line_2 - divided_line_1) // ada_train_batch_size))

# 训练好模型后保存模型
state_dict = {
    'estimators': clf.estimators,
    'alphas': clf.alphas.detach()
}
torch.save(state_dict, r'FaceRecognition\ModelParameter\adaboost_gpu_HaarBoost.pth')

# 加载预训练模型
clf = AdaBoostClassifier()
state_dict = torch.load(r'FaceRecognition\ModelParameter\adaboost_gpu_HaarBoost.pth')
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
        print('step:{}/{}, Accuracy:{}'.format(step + 1, (sample_num - divided_line_2) // test_batch_size, acc))
