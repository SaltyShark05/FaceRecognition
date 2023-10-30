import HaarClassifierDataProcess as DP
import torch

# Haar特征模板
haar_2_1_template = {
    'type': [2, 1],
    'weights': [1, -1],
}


# 根据积分图计算矩形值
def calc_rectangle_values(integral_img, feature, positions):
    # 获取haar特征模板的样式和权重
    fx, fy = feature['type']
    w1, w2 = feature['weights']
    # 获取矩形大小
    rect1_w = fx // 2
    rect1_h = fy
    rect2_w = fx - rect1_w
    rect2_h = fy
    # 获取矩形坐标
    x1, y1 = positions
    x2, y2 = x1 + rect1_w, y1 + rect1_h
    x3, y3 = x2, y1
    x4, y4 = x3 + rect2_w, y1 + rect2_h
    # 计算两个矩形的值
    s1 = integral_img[x1:x2, y1:y2]
    s2 = integral_img[x3:x4, y3:y4]
    # 加权求和得到特征值
    feature_vals = w1*s1 + w2*s2
    return feature_vals


# 计算haar模板的所有子模板(现在只局限在2*1的haar特征模板)
features = []
feature_type_x, feature_type_y = haar_2_1_template['type']
fx_scale_max = 96 // feature_type_x
fy_scale_max = 96 // feature_type_y
for fx_scale in range(1, fx_scale_max + 1):
    for fy_scale in range(1, fy_scale_max + 1):
        feature = {
            'type': [feature_type_x * fx_scale, feature_type_y * fy_scale],
            'weights': [1, -1],
        }
        features.append(feature)

all_sample_haar_features = []
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
        sample_integral_image = torch.cumsum(sample_item, dim=0).cumsum(dim=1)
        # 在进行积分图计算时需要加边进行计算
        pad_left = torch.zeros(96, 1)  # 左加边
        sample_integral_image = torch.cat([pad_left, sample_integral_image], dim=1)
        pad_top = torch.zeros(1, 97)  # 上加边
        sample_integral_image = torch.cat([pad_top, sample_integral_image], dim=0)
        sample_integral_image_items.append(sample_integral_image)

    # 计算haar特征，获得haar特征向量
    # 遍历每张图片
    for sample_integral_image in sample_integral_image_items:
        # 遍历haar特征子模板
        for i in range(len(features)):
            # 获取该haar模板的样式
            weights = features[i]['weights']
            x_feature_type, y_feature_type = features[i]['type']
            # 提前计算出所有位置
            positions = []
            x_max_steps = 96 - x_feature_type + 1
            y_max_steps = 96 - y_feature_type + 1
            for x in range(x_max_steps):
                for y in range(y_max_steps):
                    positions.append([x, y])
            # 向量化计算特征矩形值
            rectangle_values = []
            for pos in range(len(positions)):
                rv = calc_rectangle_values(sample_integral_image, features[i], positions[pos])
                rv = rv.squeeze(dim=0)
                rectangle_values.append(rv)
            haar_features_tensor = torch.cat(rectangle_values)
            print(i)
    sample_haar_features_tensor = torch.cat(rectangle_values)
all_sample_2_1_haar_features_tensor = torch.cat(sample_haar_features_tensor)
