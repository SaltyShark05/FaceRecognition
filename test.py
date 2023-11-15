import torch


def generate_haar_feature(image, size, haar_type):
    def calc_edge_1_value(sample_integral_image, features, size, positions, weights):
        # calc_edge_1_value的向量化实现
        x1, y1 = positions[:, 0], positions[:, 1]
        x2, y2 = x1 + size // features['type'][0], y1 + size // features['type'][1]
        area_l = (
            sample_integral_image[x1, y1] + sample_integral_image[x2, y2] -
            sample_integral_image[x2, y1] - sample_integral_image[x1, y2]
        )
        x3, y3 = x2, y1
        x4, y4 = x3 + size//features['type'][0], y2
        area_r = (
            sample_integral_image[x3, y3] + sample_integral_image[x4, y4] -
            sample_integral_image[x4, y3] - sample_integral_image[x3, y4]
        )
        value = (area_l * weights[0] + area_r * weights[1]) / (size * size)
        return value

    image = image.squeeze(dim=0)
    sample_integral_image = torch.cumsum(image, dim=0).cumsum(dim=1)

    # 对积分图进行填充
    sample_integral_image = torch.nn.functional.pad(sample_integral_image, (1, 0, 1, 0))

    if haar_type == 'edge_1':
        haar_template = {'type': [2, 1], 'weights': [1, -1]}
    elif haar_type == 'line_1':
        haar_template = {'type': [3, 1], 'weights': [-0.5, 1, -0.5]}
    elif haar_type == 'center_1':
        haar_template = {'type': [2, 2], 'weights': [0.5, -0.5, -0.5, 0.5]}

    weights = haar_template['weights']
    max_steps = 48 - size + 1

    positions = torch.tensor([[x, y] for x in range(0, max_steps, 4) for y in range(0, max_steps, 4)])

    if haar_type == 'edge_1':
        haar_features = calc_edge_1_value(sample_integral_image, haar_template, size, positions, weights)

    return torch.tensor(haar_features)


# 例子用法：
image = torch.randn(1, 48, 48)
size = 6
haar_type = 'edge_1'
result = generate_haar_feature(image, size, haar_type)
print(result)