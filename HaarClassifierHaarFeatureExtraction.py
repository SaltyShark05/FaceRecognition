import HaarClassifierDataProcess as DP
import torch

# Haar特征模板
haar_2_1_template = {
    'type': [2, 1],
    'position': [0, 0],
    'size': [1, 1],
    'weights': [1, -1],
}

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
    # 遍历haar特征子模板
    for i in range(len(features)):
        x_feature_type, y_feature_type = features[i]['type']
        x_max_steps = 96 - x_feature_type + 1
        y_max_steps = 96 - y_feature_type + 1
        # haar特征子模板固定，移动haar特征子模板的位置
        for x in range(x_max_steps):
            for y in range(y_max_steps):
                
                # pos = features[i]['pos']
                # size = features[i]['size']
                # weights = features[i]['weights']

                # # 计算特征值
                # value = calc_rect_value(integral_img, pos, size, weights)

                # # 存储结果
                # results[i, x, y] = value
                
    # for sample_integral_image_item in sample_integral_image_items:
    #     haar_feature = 1
    #     feature_list = []
    #     feature_list.append(haar_feature)
    #     feature_tensor = torch.Tensor(feature_list)

    # print("step:{}, sample_transform:{}, sample_label:{}".format(step, sample_transform, sample_label))
