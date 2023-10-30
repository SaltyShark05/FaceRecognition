import HaarClassifierDataProcess as DP
import torch

# Haar特征模板
haar_2_1_template = {
    'type': [2, 1],
    'position': [0, 0],
    'size': [1, 1],
    'weights': [1, -1],
}

for step, (sample_transform, sample_label) in enumerate(DP.Face_dataloader):
    # 将数据拆分，降维成二维Tensor，合并成列表
    sample_divideds = torch.chunk(sample_transform, chunks=10, dim=0)
    sample_items = []
    for sample_item in sample_divideds:
        sample_item = sample_item.squeeze(dim=0)
        sample_item = sample_item.squeeze(dim=0)
        sample_items.append(sample_item)

    # 计算积分图
    sample_integral_image_items = []
    for sample_item in sample_items:
        sample_integral_image = torch.cumsum(sample_item, dim=0)
        sample_integral_image = torch.cumsum(sample_integral_image, dim=1)
        sample_integral_image_items.append(sample_integral_image)

    # 计算haar特征，获得haar特征向量
    # 计算haar模板的所有子模板
    features = [haar_2_1_template]
    # 遍历特征和位置
    for i in range(len(features)):
        for x in range(96):
            for y in range(96):
                pos = features[i]['pos']  
                size = features[i]['size']
                weights = features[i]['weights']
                
                # 计算特征值
                value = calc_rect_value(integral_img, pos, size, weights)  
                
                # 存储结果
                results[i, x, y] = value
    for sample_integral_image_item in sample_integral_image_items:
        haar_feature = 1
        feature_list = []
        feature_list.append(haar_feature)
        feature_tensor = torch.Tensor(feature_list)

    # print("step:{}, sample_transform:{}, sample_label:{}".format(step, sample_transform, sample_label))
