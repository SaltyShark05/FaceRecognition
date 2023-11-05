import torch
weights = torch.load('adaboost.pth')
weights = {k: v for k, v in weights.items()}
print(weights)
