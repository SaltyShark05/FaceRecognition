import torch
weights = torch.load(r'FaceRecognition\ModelParameter\adaboost.pth')
weights = {k: v for k, v in weights.items()}
print(weights)
