import DataProcess as DP


imgs_batch, labels_batch = next(iter(DP.train_dl))
print(imgs_batch.shape)

# 将数据分割成二维Tensor
for i in range():
    imgs = imgs_batch.split([1,max-i] dim=0)