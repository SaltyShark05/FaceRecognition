import HaarClassifierDataProcess as DP
import torch

# Haar特征模板
haar_2_1_template = {
    'type': [2, 1],
    'position': [0, 0],
    'size': [1, 1],
    'weights': [1, -1],
}


# 根据积分图计算矩形值
def calc_rectangle_value(sample_integral_image, pos_x, pos_y, size_w, size_h):
    value = sample_integral_image[pos_x, pos_y] + sample_integral_image[pos_x + size_w, pos_y + size_h] - sample_integral_image[pos_x + size_w, pos_y] - sample_integral_image[pos_x, pos_y + size_h]
    return value


# 计算haar模板的所有子模板(现在只局限在2*1的haar特征模板)
features = []
feature_type_x, feature_type_y = haar_2_1_template['type']
feature_size_w, feature_size_h = haar_2_1_template['size']
fx_scale_max = 96 // feature_type_x
fy_scale_max = 96 // feature_type_y
for fx_scale in range(1, fx_scale_max + 1):
    for fy_scale in range(1, fy_scale_max + 1):
        current_type_x = feature_type_x * fx_scale
        current_type_y = feature_type_y * fy_scale
        current_size_w = feature_size_w * fx_scale
        current_size_h = feature_size_h * fy_scale
        feature = {
            'type': [current_type_x, current_type_y],
            'position': [0, 0],
            'size': [current_size_w, current_size_h],
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
        sample_integral_image = torch.cumsum(sample_item, dim=0)
        sample_integral_image = torch.cumsum(sample_integral_image, dim=1)
        # 在进行积分图计算时需要加边进行计算
        # 左加边
        pad_left = torch.zeros(96, 1)
        sample_integral_image = torch.cat([pad_left, sample_integral_image], dim=1)
        # 上加边
        pad_top = torch.zeros(1, 97)
        sample_integral_image = torch.cat([pad_top, sample_integral_image], dim=0)
        sample_integral_image_items.append(sample_integral_image)

    # 计算haar特征，获得haar特征向量
    sample_haar_features = []
    # 遍历每张图片
    for sample_integral_image in sample_integral_image_items:
        # 遍历haar特征子模板
        haar_features = []
        for i in range(len(features)):
            x_feature_type, y_feature_type = features[i]['type']
            x_max_steps = 96 - x_feature_type + 1
            y_max_steps = 96 - y_feature_type + 1
            # haar特征子模板固定，移动haar特征子模板的位置
            pos_x, pos_y = features[i]['position']
            size_w, size_h = features[i]['size']
            for x in range(x_max_steps):
                for y in range(y_max_steps):
                    current_pos_x = pos_x + x
                    current_pos_y = pos_y + y
                    area_left = calc_rectangle_value(sample_integral_image, current_pos_x, current_pos_y, size_w, size_h)
                    area_right = calc_rectangle_value(sample_integral_image, current_pos_x + size_w, current_pos_y, size_w, size_h)
                    haar_feature_value = area_left * features[i]['weights'][0] + area_right * features[i]['weights'][1]
                    haar_features.append(haar_feature_value)
        haar_features_tensor = torch.tensor(haar_features)
        sample_haar_features.append(haar_features_tensor)
    sample_haar_features_tensor = torch.tensor(sample_haar_features)
    all_sample_haar_features.append(sample_haar_features_tensor)
    print("step:{}".format(step))
all_sample_haar_features_tensor = torch.tensor(all_sample_haar_features)
