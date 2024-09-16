
import torchvision.transforms as transforms
from model.Unet import UNet
from model.LeViT_UNeT import Build_LeViT_UNet_192
from model.doubleunet_pytorch import build_doubleunet
from model.NestedUNet import NestedUNet
from model.ResUnetplus import ResUnetPlusPlus
from model.Resunet import ResUnet
# from model.self_Resunet_H import UResnet_H
# from model.doubleResUNet import UResnet_H
from model.DCSAU_Net1 import Model
from model.Res18UNet import Resnet34_Unet
from model.CS_CO_E import UResnet_E
# from model.CS_CO_H import UResnet_H
from model.PUNet import PUNet
from model.NucleiSegNet import nuclei_segnet
from model.atten_UNet import AttU_Net
from model.TranSEFusionNet import TransattU_Net
#from model.improvedUnet import PUNet
import torch.nn as nn
import torch.nn.functional as F
from dataset import ISBI_Loader
from dataset2 import ISBI_Loader2
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import tqdm
import copy



seed = 0 #yuan3

#loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1

        probs = logits
        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        #IoU loss
        IoU = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) -intersection.sum(1)  + smooth)
        IoU = 1 - IoU.sum() / bs

        return score

class DiceLoss1(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss1, self).__init__()

    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1

        probs = logits
        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = score.sum() / bs

        return score

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):

        bs = target.size(0)
        m1 = x.view(bs, -1)
        m2 = target.view(bs, -1)
        p = copy.copy(m1)
        p=torch.where(p>0.5,1,0)

        m3 = ~(p.int() ^ m2.int())+2

        index = torch.where(m2.bool(),self.m_list[1],self.m_list[0])
        index = index.type(torch.cuda.FloatTensor)
        x_m = m1- index #   1. x_m: j == y  ;  2. x:j != y;

        output = torch.where(m3.bool(), x_m, m1)

        result = nn.BCELoss()( F.sigmoid(output), m2).double()
        return  result


class LMFLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        dice = DiceLoss()(input,target).double()
        # bce_loss = nn.BCELoss()(pred, truth).double()

        cls_num_list = []
        for i in range(2):
            cls_num_list.append(truth.tolist().count(i))

        forcal_loss = focal_loss(pred, truth).double()
        LDAM = LDAMLoss(cls_num_list =cls_num_list)(input,target).double()

        return LDAM


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)

def valid_net(net,datapath,criterion,epoch):
    losses = 0
    valid_loader = torch.utils.data.DataLoader(dataset=datapath,
                                               batch_size=1,
                                               shuffle=True,
                                               drop_last =True)
    p_bar = tqdm.tqdm(valid_loader)
    for i, (image, label) in enumerate(p_bar):
        # 将数据拷贝到device中
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        pred = net(image)
        loss = criterion(pred, label)
        p_bar.set_description('Epoch {}'.format(epoch))

        p_bar.set_postfix(loss=loss.item())
        losses += loss.item()
    return losses/(i+1)

def train_net(name3,net, device, data_path, epochs=40, batch_size=8, lr=0.0001):
    # 加载训练集
    np.random.seed(seed)
    #np.random.seed(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    txtpath = name3 +'.txt'
    file = open(txtpath,"a")

    isbi_dataset = ISBI_Loader(data_path=data_path)
    isbi2_dataset = ISBI_Loader2(data_path2)


    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=True )

    # 定义RMSprop算法
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),lr=lr, weight_decay=1e-8)
    # 定义Loss算法

    # criterion = LMFLoss()/home/student/Data_disk/paper1_res/percent/MoNu/ResUnet
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        losses =0
        i = 0
        p_bar = tqdm.tqdm(train_loader)

        for i,(image, label) in enumerate(p_bar):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 使用网络参数，输出预测结果
            pred = net(image)

            loss = criterion(pred, label)
            # print(epoch, ':Loss/train', loss.item())
            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss.item())

            # 更新参数
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print('Epoch{:3d}--train_loss{:.4f}-'.format(epoch,  losses/(i+1)))
        net.eval()
        valid_loss = valid_net(net, isbi2_dataset, criterion, epoch)
        print('Epoch{:3d}--vaild_loss{:.4f}-'.format(epoch, valid_loss))

        file.write(str(epoch) + ' train_loss:' + str(losses / (i + 1)) + ' '+ 'valid_loss:'+str(valid_loss) + '\n')

        # 保存loss值最小的网络参数
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("save")
            torch.save(net.state_dict(), weight_path + 'best_model' + str(epoch) + '.pth')

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("cuda is oks")
    else:
        print("cuda is not")
    # 加载网络，图片单通道1，分类为1。

    file = 'CoN'

    weight_path = '/home/student/Data_disk/bucongshiyan/REs/'+file+'/'

    # 指定训练集地址，开始训练   CoNSeP_dataset/SEG_dataset_test
    data_path = "/home/student/Data_disk/part1_dataset/Train100/"
    data_path2 = "/home/student/Data_disk/part1_dataset/test/CoNSep/"
    # train_net(name2,net, device, data_path)
    # path = "/home/student/PycharmProjects/pycharm/liu/liu/CS-CO-main/ablation/checkpoint/gamma2_10.0_0.001/csco_cs-co_Adam-no_None_10_0.001_1e-06_1.0_45_0.04373.pth"
    # path = "/home/student/PycharmProjects/pycharm/liu/liu/CS-CO-main/checkpoint/NCT_CRC/csco/csco_cs_Adam-step_None_16_0.001_0.0_1.0_1_0.08541.pth"
    # path = '/home/student/PycharmProjects/pycharm/liu/cell/seg/Unet/save/LeViT/best_model11.pth'
    name3 = weight_path+file
    print(name3)
    # net = NestedUNet(num_classes=1)
    net = ResUnet(1)
    #net = PUNet(n_channels=1, n_classes=1)
    # net =  AttU_Net(img_ch=1, output_ch=1)
    # net = nuclei_segnet(input=1, num_class=1)
    # net = UResnet_H()
    # net = UNet(n_channels=1, n_classes=1)
    # net = TransattU_Net(num_classes=1,input_channels=1)
    #net = Resnet34_Unet(in_channel=1, out_channel=1, pretrained=True)
    #net = Build_LeViT_UNet_192(num_classes=1)
    # net = Model(img_channels=1, n_classes=1)
    # net = build_doubleunet()
    # net = ResUnetPlusPlus(1)
    # net = UNet(n_channels=1, n_classes=1)
    # net.load_state_dict(torch.load(path), strict=False)
    net.to(device=device)
    cls_state_dict = net.state_dict()
    for name, param in net.named_parameters():
    #     if 'he_decoder'  not in name:
    #         if 'bridge' not in name:
    #             if 'output_layer' not in name:
    #                 param.requires_grad = False
        print(str(name) + "------->" + str(param.requires_grad))

    train_net(name3,net, device, data_path)
