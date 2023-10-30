import HaarClassifierDataProcess as DP
import torch
device = torch.device("cuda") 

# 模板定义
haar_template = {
    'type': [2, 1],
    'weights': [1, -1],
}

all_features = []
for sample in DP.Face_dataloader:

  imgs = sample[0].to(device)

  # 积分图
  integral_imgs = torch.cumsum(imgs, dim=1).cumsum(dim=2).to(device)  

  # 位置坐标
  pos = torch.cartesian_prod(torch.arange(max_x), torch.arange(max_y)).t().to(device)

  # 特征矩形 
  rects = get_rect_coords(pos, features).to(device)

  # 特征值
  rect_vals = integral_imgs[:, rects[...,0], rects[...,1]]
  feat_vals = torch.matmul(weights.to(device), rect_vals)

  all_features.append(feat_vals)

all_features = torch.cat(all_features).to(device) 

# 模型和优化器
model = model.to(device)
optimizer = optimizer.to(device)

# 训练循环
for x, y in dataloader:
  x = x.to(device)
  y = y.to(device)
  
  preds = model(x)
  loss = loss_fn(preds, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()