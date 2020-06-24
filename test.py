import os
from hyper import *
from model.CPD_models import CPD_VGG
from model.CPD_models import CPD_3DResNet
from dataset import *
import cv2
import torch
import torch.nn.functional as F

max = 600
name = 'model_cpd_250616_Adam_1e-05_20_BCEWithLogitsLoss_8.pth'
checkpoint = './checkpoint/cpd_0616/' + name
# checkpoint = './pre-trained/audio/model_cpd_300525_Adam_1.0000000000000002e-07_20_BCEWithLogitsLoss_29.pth'
# checkpoint = 'C:/Users/MSI-PC/Desktop/cpd_wrs_2.21_导入参数后效果很好/train_weight/vgg_dfh1k_cpd/vgg.pth.10'
model = CPD_3DResNet()
model.load_state_dict(torch.load(checkpoint))

print('\n', name, '\n')
'''
save_model = torch.load(checkpoint)
now_model = model.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in now_model.keys()}
now_model.update(state_dict)
model.load_state_dict(now_model)
'''

model.cuda()
model.eval()

dir_num = 0
img_num = 0

for i in os.listdir(test_data_path):
    test_frames = test_data_path + i + '/'
    save_path = './test_result/' + checkpoint.split('/')[-1].split('.pth')[0] + '/' + i + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = LoadTestData(test_frames)

    for j in range(test_loader.size):
        image, name, index = test_loader.load_data()

        image = image.reshape(1, image.size(0), image.size(1), image.size(2))
        image = image.cuda()

        ress = model(image)
        h = 256
        w = 320
        
        res = F.interpolate(ress, size=[h, w], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res * 255)
        img_num += 1
        '''
        if j == 150:
         break
        '''
    print("===== {} finished !=====\n".format(i))
    dir_num += 1
    if dir_num == max:
        exit(0)

print("Total test images == {} == !\n".format(img_num))


