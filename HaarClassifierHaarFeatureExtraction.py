import FaceRecognition.HaarClassifier as DP
import torch

# Haar特征模板
haar_2_1_template = {
    'type': [2, 1],
    'size': [6, 6],
    'scale': [2, 3, 4, 5, 6, 7, 8],
    'weights': [1, -1],
}


# 根据积分图计算矩形值
def calc_rectangle_value(sample_integral_image, features, position, weights):
    feature_type_x, feature_type_y = features['type']
    feature_size_w, feature_size_h = features['size']
    feature_weight_l, feature_weight_r = weights
    x1, y1 = position
    x2, y2 = x1 + feature_size_w//feature_type_x, y1 + feature_size_h//feature_type_y
    x3, y3 = x2, y1
    x4, y4 = x3 + feature_size_w//feature_type_x, y2
    area_l = sample_integral_image[x1, y1] + sample_integral_image[x2, y2] - sample_integral_image[x2, y1] - sample_integral_image[x1, y2]
    area_r = sample_integral_image[x3, y3] + sample_integral_image[x4, y4] - sample_integral_image[x4, y3] - sample_integral_image[x3, y4]
    value = area_l * feature_weight_l + area_r * feature_weight_r
    return value


# 计算haar模板的所有子模板(现在只局限在2*1的haar特征模板)
features = []
feature_type_x, feature_type_y = haar_2_1_template['type']
feature_scale = haar_2_1_template['scale']
feature_size_w, feature_size_h = haar_2_1_template['size']
for scale in feature_scale:
    feature = {
        'type': [2, 1],
        'size': [feature_size_w * scale, feature_size_h * scale],
        'weights': [1, -1],
    }
    features.append(feature)
print(features)

all_sample_haar_features = []
for step, (sample_transform, sample_label) in enumerate(DP.train_dl):
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
        sample_integral_image = torch.cumsum(sample_item, dim=0).cumsum(dim=1)
        # 在进行积分图计算时需要加边进行计算
        pad_left = torch.zeros(96, 1)  # 左加边
        sample_integral_image = torch.cat([pad_left, sample_integral_image], dim=1)
        pad_top = torch.zeros(1, 97)  # 上加边
        sample_integral_image = torch.cat([pad_top, sample_integral_image], dim=0)
        sample_integral_image_items.append(sample_integral_image)

    # 计算haar特征，获得haar特征向量
    # 遍历每张图片
    sample_haar_features = []
    for sample_integral_image in sample_integral_image_items:
        # 遍历haar特征子模板
        for i in range(len(features)):
            # 获取该haar模板的样式
            feature_type_x, feature_type_y = features[i]['type']
            feature_size_w, feature_size_h = features[i]['size']
            weights = features[i]['weights']
            # 提前计算出所有位置
            positions = []
            x_max_steps = 96 - feature_size_w + 1
            y_max_steps = 96 - feature_size_h + 1
            for x in range(x_max_steps):
                for y in range(y_max_steps):
                    positions.append([x, y])
            # 计算特征矩形值
            haar_features = []
            for pos in range(len(positions)):
                haar_feature = calc_rectangle_value(sample_integral_image, features[i], positions[pos], weights)
                haar_features.append(haar_feature)
            haar_features_tensor = torch.tensor(haar_features)
            print(haar_features_tensor.shape)
        # sample_haar_features.append(haar_features_tensor)
#     sample_haar_features_tensor = torch.tensor(sample_haar_features)
#     print("sample_haar_features_tensor:{}".format(sample_haar_features_tensor))
#     all_sample_haar_features.append(sample_haar_features_tensor)
# all_sample_2_1_haar_features_tensor = torch.tensor(all_sample_haar_features)
# print("all_sample_2_1_haar_features_tensor:{}".format(all_sample_2_1_haar_features_tensor))
