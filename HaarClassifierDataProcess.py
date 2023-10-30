import glob
import torch
from PIL import Image
from torchvision import transforms


# 正样本加载，返回文件的路径，以list存储
positive_sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\ClassifierTrainingData\Positive\*.bmp')

# 负样本加载，返回文件的路径，以list存储
negative_sample_path = glob.glob(r'E:\VScode\Python\FaceRecognition\ClassifierTrainingData\Negative\*.jpg')

# 创建样本路径，正负样本合并
all_sample_path = positive_sample_path + negative_sample_path

# 创建样本标签
all_sample_label = []

for img in positive_sample_path:
    all_sample_label.append(0)

for img in negative_sample_path:
    all_sample_label.append(1)

# 对数据进行转换处理
all_sample_transform = transforms.Compose([
                transforms.Resize((96, 96)),  # 做的第一步转换，统一大小
                transforms.Grayscale(),  # 转换为灰度图像
                transforms.ToTensor()  # 第二步转换，作用：第一把PIL的Image转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])


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
        return sample_transform, sample_label

    # 返回长度
    def __len__(self):
        return len(self.sample_path)


BATCH_SIZE = 10
Face_dataset = Mydataset(all_sample_path, all_sample_label, all_sample_transform)
Face_dataloader = torch.utils.data.DataLoader(
                            Face_dataset,
                            batch_size=BATCH_SIZE,  # 抽出每一小组的大小
                            shuffle=True  # 样本集中的数据进行打乱
)
