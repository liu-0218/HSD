import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as transforms
import random

#train

idea = 'image'  # H E Tissue Images
class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, idea+'/*.tif'))

    def augment(self, image,flipCode):
        flip = cv2.flip(image,flipCode)  #1 0 -1
        return flip

    def augment1(self, image,a):
        filp = cv2.rotate(image,a)
        return filp

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 根据image_path生成label_path
        label_path = image_path.replace('/'+idea+'/', '/label/')
        # label_path = label_path.replace('.tif', '.tif')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # image.Normalize(mean=[0.610,0.430,0.590],std=[0.235,0.239,0.190])


        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # a = random.choice([cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, 2])
        # if flipCode != 2  :
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        # if a != 2:
        #     image = self.augment1(image, a)
        #     label = self.augment1(label, a)


        a = 224
        b= 1

        image = cv2.resize(image, (a, a))
        label = cv2.resize(label, (a, a))
        image = image.reshape(b, a, a)
        label = label.reshape(1, a, a)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("/home/student/PycharmProjects/pycharm/liu/cell/seg/data/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)