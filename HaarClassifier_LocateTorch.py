import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
import os

# 预处理后的图片大小
target_height = 1024
target_width = 1024


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.pnet = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=10, init=0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=16, init=0.25),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=32, init=0.25)
        )

        self.classifier = nn.Conv2d(32, 2, kernel_size=1)
        self.bbox_regress = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.pnet(x)

        classifier = F.softmax(self.classifier(x), dim=1)
        bbox_regress = self.bbox_regress(x)
        return classifier, bbox_regress


class RNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.rnet = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=28, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=48, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0),
            nn.PReLU(num_parameters=64, init=0.25),

            nn.Permute((3, 2, 1))
        )

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(576, 128)
        self.prelu1 = nn.PReLU(num_parameters=128, init=0.25)

        self.classifier = nn.Linear(128, 2)
        self.bbox_regress = nn.Linear(128, 4)

    def forward(self, x):
        x = self.rnet(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.prelu1(x)

        classifier = F.softmax(self.classifier(x), dim=1)
        bbox_regress = self.bbox_regress(x)
        return classifier, bbox_regress


class ONet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.onet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=32, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=64, init=0.25),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(num_parameters=64, init=0.25),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.PReLU(num_parameters=128, init=0.25),
            nn.Permute((3, 2, 1))
        )

        self.flatten = nn.Flatten()
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(num_parameters=256, init=0.25)

        self.classifier = nn.Linear(256, 2)
        self.bbox_regress = nn.Linear(256, 4)
        self.landmark_regress = nn.Linear(256, 10)

    def forward(self, x):
        x = self.onet(x)

        x = self.flatten(x)
        x = self.dense5(x)
        x = self.prelu5(x)

        classifier = F.softmax(self.classifier(x), dim=1)
        bbox_regress = self.bbox_regress(x)
        landmark_regress = self.landmark_regress(x)

        return classifier, bbox_regress, landmark_regress


class MTCNN(nn.Module):
    def __init__(self):
        super(MTCNN, self).__init__()
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, x):
        # PNet
        pnet_cls, pnet_bbox = self.pnet(x)
        pnet_cls_prob = F.softmax(pnet_cls, dim=1)[:, 1, :, :]  # 取softmax后的第二个通道，表示人脸的概率

        # 选择概率大于阈值的候选框
        pnet_mask = (pnet_cls_prob > threshold).float()
        pnet_bbox *= pnet_mask.unsqueeze(dim=1)

        # RNet
        rnet_cls, rnet_bbox = self.rnet(x, pnet_bbox)
        rnet_cls_prob = F.softmax(rnet_cls, dim=1)[:, 1, :, :]

        # 选择概率大于阈值的候选框
        rnet_mask = (rnet_cls_prob > threshold).float()
        rnet_bbox *= rnet_mask.unsqueeze(dim=1)

        # ONet
        onet_cls, onet_bbox, onet_pts = self.onet(x, rnet_bbox)
        onet_cls_prob = F.softmax(onet_cls, dim=1)[:, 1, :, :]

        # 选择概率大于阈值的候选框
        onet_mask = (onet_cls_prob > threshold).float()
        onet_bbox *= onet_mask.unsqueeze(dim=1)

        return pnet_bbox, rnet_bbox, onet_bbox


# 标签处理
def label_process(file_path):
    # 创建样本标签
    num_of_face_labels = []
    bboxs_labels = []

    # 读取文件的所有行
    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    line_index = 0
    while (line_index < total_lines):
        # 获取图片路径与缩放比例
        line = lines[line_index].strip()
        file_name = line
        sample_path = 'FaceRecognition/MTCNNTrainingData/WIDER_train/images/'
        sample_path = os.path.join(sample_path, file_name)

        # 计算调整比例
        pil_img = Image.open(sample_path)
        original_width, original_height = pil_img.size
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # 获取人脸数量
        line_index += 1
        line = lines[line_index].strip()
        data = line.split()
        num_of_face = int(data[0])

        # 人脸数量为空判断
        if (num_of_face == 0):
            next_lines = 1
        else:
            next_lines = num_of_face

        # 获取人脸位置
        bboxs = []
        for _ in range(next_lines):
            line_index += 1
            line = lines[line_index].strip()
            data = line.split()
            bbox = list(map(int, data[0:4]))
            adjusted_bbox = [
                int(bbox[0] * scale_x),  # 调整 x 坐标
                int(bbox[1] * scale_y),  # 调整 y 坐标
                int(bbox[2] * scale_x),  # 调整宽度
                int(bbox[3] * scale_y),  # 调整高度
            ]
            bboxs.append(adjusted_bbox)

        # 添加到标签
        num_of_face_tensor = torch.tensor(num_of_face)
        bboxs_tensor = torch.tensor(bboxs)

        num_of_face_labels.append(num_of_face_tensor)
        bboxs_labels.append(bboxs_tensor)
        line_index += 1

    return num_of_face_labels, bboxs_labels


# 数据处理
def data_process():
    # 获取样本路径
    sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\MTCNNTrainingData\WIDER_train\images\*\*.jpg')

    # 获取样本图像
    sample_transform = []
    sample_transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor()
    ])

    # 获取标签
    file_path = r'FaceRecognition\MTCNNTrainingData\wider_face_split\wider_face_train_bbx_gt.txt'
    num_of_face_labels, bboxs_labels = label_process(file_path)

    class Mydataset(torch.utils.data.Dataset):
        # 类初始化
        def __init__(self, sample_path, num_of_face_labels, bboxs_labels, sample_transform):
            self.sample_path = sample_path
            self.num_of_face_labels = num_of_face_labels
            self.bboxs_labels = bboxs_labels
            self.sample_transform = sample_transform

        # 进行切片
        def __getitem__(self, index):                # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
            sample_path = self.sample_path[index]
            pil_img = Image.open(sample_path)
            sample_transform = self.sample_transform(pil_img)
            sample_num_of_face_label = self.num_of_face_labels[index]
            bboxs_labels = self.bboxs_labels[index]
            # 返回特征和标签
            return sample_transform, sample_num_of_face_label, bboxs_labels

        # 返回长度
        def __len__(self):
            return len(self.sample_path)

    index = np.random.permutation(len(sample_path))
    sample_path = np.array(sample_path)[index]

    train_dataset = Mydataset(sample_path, num_of_face_labels, bboxs_labels, sample_transform)
    train_batch_size = 10
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    return train_dataloader


# PNet的训练
def pnet_train(pnet_train_dataloader):
    pnet = PNet()
    pnet.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(pnet.parameters(), lr=0.001)

    # 训练循环
    epochs = 10
    for epoch in range(epochs):
        for step, (data, num_of_face_label, sample_bboxs_label) in enumerate(pnet_train_dataloader):
            # GPU加速
            data = data.to(device)
            num_of_face_label = num_of_face_label.to(device)
            sample_bboxs_label = sample_bboxs_label.to(device)
            optimizer.zero_grad()
            num_of_face_output, sample_bboxs_output = pnet(data)
            # 计算损失
            loss = criterion(sample_bboxs_output, sample_bboxs_label)  # 需要根据网络输出和任务计算损失
            loss.backward()
            optimizer.step()
            print('epoch:{}/{}, step:{}, loss:{}'.format(epoch + 1, epochs, step + 1, loss))
        print('epoch finish')


if __name__ == "__main__":
    # GPU加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据处理
    train_dataloader = data_process()

    # PNet的训练
    pnet_train(train_dataloader)
