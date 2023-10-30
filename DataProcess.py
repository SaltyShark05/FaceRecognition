import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root):
        self.imgs_path = root

    # 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    # 返回长度
    def __len__(self):
        return len(self.imgs_path)


# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'E:\VScode\Python\FaceRecognition\Data\*\*.jpg')  # 返回符合path格式的所有文件的路径，以list存储返回
# 循环遍历输出列表中的每个元素，显示出每个图片的路径
'''
for var in all_imgs_path:
    print(var)
'''


# 利用自定义类Mydataset创建对象weather_dataset
Face_dataset = Mydataset(all_imgs_path)
'''
print(len(Face_dataset))  # 返回文件夹中图片总个数
print(Face_dataset[5:7])  # 切片，显示第6张、第7张图片，python左闭右开
'''
Face_dataloader = torch.utils.data.DataLoader(Face_dataset, batch_size=2)  # 每次抽出2个样本
'''
print(next(iter(Face_dataloader)))  # 打印出第一次抽出的样本
'''

species = ['WJ', 'DC']
# i是序号，c是spcies的内容，enumerate在进行迭代，dict（c,i）按c：i顺序生成字典
species_to_id = dict((c, i) for i, c in enumerate(species))
'''
print(species_to_id)
'''
# 反转字典
id_to_species = dict((v, k) for k, v in species_to_id.items())
'''
print(id_to_species)
'''
all_labels = []
# 对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
'''
print(all_labels)  # 得到所有标签
'''

# 对数据进行转换处理
transform = transforms.Compose([
                transforms.Resize((96, 96)),  # 做的第一步转换，统一大小
                transforms.Grayscale(),  # 转换为灰度图像
                transforms.ToTensor()  # 第二步转换，作用：第一把PIL的Image转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
])


class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):                # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)                 # p ip install pillow
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 10
Face_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
Face_dataloader = data.DataLoader(
                            Face_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True  # 样本集中的数据有进行打乱
)

imgs_batch, labels_batch = next(iter(Face_dataloader))
'''
print(imgs_batch)
print(labels_batch)
print(imgs_batch.shape)
'''


'''
plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i+1)
    plt.title(id_to_species.get(label.item()))
    plt.imshow(img)
plt.show()
'''

# 划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

# 80% 的数据作为训练集
s = int(len(all_imgs_path)*0.8)
'''
print(s)
'''

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

train_ds = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_ds = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TrainSet Labels
test_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TestSet Labels
