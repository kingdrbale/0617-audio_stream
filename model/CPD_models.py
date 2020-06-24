import torch
import torch.nn as nn
import torchvision
from hyper import *
from model.HolisticAttention import HA
from model.ResNe3D import resnet18
from model.ResNet_18 import ResNet2_18
from model.vgg import B2_VGG
from model.v_a_con import *


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x1, x2, x3):
        # x1: 1/16 x2: 1/8 x3: 1/4
        x1_1 = x1

        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)

        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class CPD_VGG(nn.Module):
    def __init__(self, channel=32):
        super(CPD_VGG, self).__init__()
        self.vgg = B2_VGG()
        self.rfb3_1 = RFB(256, channel)
        self.rfb4_1 = RFB(512, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(256, channel)
        self.rfb4_2 = RFB(512, channel)
        self.rfb5_2 = RFB(512, channel)
        self.agg2 = aggregation(channel)

        self.HA = HA()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x3_1 = x3

        x4_1 = self.vgg.conv4_1(x3_1)
        x5_1 = self.vgg.conv5_1(x4_1)

        '''
            Audio Stream
        '''
        y1 = self.spec_vgg.conv1(y)
        y2 = self.spec_vgg.conv2(y1)
        y3 = self.spec_vgg.conv3(y2)
        y3_1 = y3

        y4_1 = self.spec_vgg.conv4_1(y3_1)
        y5_1 = self.spec_vgg.conv5_1(y4_1)
        '''
            video stream and audio stream concentration 
        '''
        print("CPD video : ", x3_1.shape, x4_1.shape, x5_1.shape)
        print("CPD audio : ", y3_1.shape, y4_1.shape, y5_1.shape)
        exit(0)
        mix3 = self.v_a_con(x3_1, y3_1)
        mix4 = self.v_a_con(x4_1, y4_1)
        mix5 = self.v_a_con(x5_1, y5_1)
        x3_1 = self.v_a_con.conv_3(mix3)
        x4_1 = self.v_a_con.conv_4(mix4)
        x5_1 = self.v_a_con.conv_5(mix5)
        '''
        print('Conca : ', mix3.shape, mix4.shape, mix5.shape)
        print('Conca : ', x3_1.shape, x4_1.shape, x5_1.shape)
        exit(0)
        '''
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)

        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), x3)
        x4_2 = self.vgg.conv4_2(x3_2)
        x5_2 = self.vgg.conv5_2(x4_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        return self.upsample(attention), self.upsample(detection)


class CPD_3DResNet(nn.Module):
    def __init__(self, channel=32):
        super(CPD_3DResNet, self).__init__()
        # self.video_branch = resnet18(shortcut_type='A', sample_size=112, sample_duration=16, last_fc=False,
        #                              last_pool=False)
        # self.audio_branch = resnet18(shortcut_type='A', sample_size=64, sample_duration=16, num_classes=12,
        #                              last_fc=False, last_pool=True)
        # self.audio_branch = ResNet2_18()
        self.audio_branch = torchvision.models.resnet18(pretrained=False)

        self.v_a_con = Video_Audio_Con()

        self.vgg = B2_VGG()
        self.rfb3_1 = RFB(256, channel)
        self.rfb4_1 = RFB(512, channel)
        self.rfb5_1 = RFB(512, channel)
        self.agg1 = aggregation(channel)

        self.rfb3_2 = RFB(256, channel)
        self.rfb4_2 = RFB(512, channel)
        self.rfb5_2 = RFB(512, channel)
        self.agg2 = aggregation(channel)

        self.HA = HA()

        self.upsample_a1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_a2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_a3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        '''
        '''
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, a):
        # f_num = 3, [bs, 128, 32, 40] [bs, 256, 16, 20] [bs, 512, 8, 10]
        # v1, v2, v3 = self.video_branch(v)
        a1, a2, a3 = self.audio_branch(a)

        '''
        a1 = self.v_a_con_audio.upsample_a1(a1)
        detection = self.v_a_con_audio.conv_last(a1)
        '''
        '''
        print(a1.shape, a2.shape, a3.shape)
        exit(0)
        '''

        '''
        a1 = torch.squeeze(a1, dim=2)
        a1 = self.v_a_con_10.upsample_a1(a1)
        detection = self.v_a_con_10.conv_last(a1)
        '''

        a1 = self.upsample_a1(a1)
        a2 = self.upsample_a2(a2)
        a3 = self.upsample_a3(a3)
        
        x3_1 = self.v_a_con.conv_3(a1)
        x4_1 = self.v_a_con.conv_4(a2)
        x5_1 = self.v_a_con.conv_5(a3)

        xx = x3_1
        # print(x3_1.shape, x4_1.shape, x5_1.shape)

        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        x5_1 = self.rfb5_1(x5_1)
        # print(x3_1.shape, x4_1.shape, x5_1.shape)

        attention = self.agg1(x5_1, x4_1, x3_1)

        x3_2 = self.HA(attention.sigmoid(), xx)
        x4_2 = self.vgg.conv4_2(x3_2)
        x5_2 = self.vgg.conv5_2(x4_2)
        # print(x3_2.shape, x4_2.shape, x5_2.shape)
        # exit(0)

        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        detection = self.agg2(x5_2, x4_2, x3_2)

        '''
        '''
        return self.upsample(detection)
