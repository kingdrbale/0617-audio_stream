import torch
import torch.nn as nn


class Video_Audio_Con(nn.Module):
    def __init__(self):
        super(Video_Audio_Con, self).__init__()

        # f_num = 10 384
        self.conv_3 = nn.Conv2d(128, 256, 1, padding=0)
        self.conv_4 = nn.Conv2d(256, 512, 1, padding=0)
        self.conv_5 = nn.Conv2d(512, 512, 1, padding=0)

    def forward(self, x, y):
        mix = torch.cat((x, y), 1)

        return mix


class Video_Audio_Con_10(nn.Module):
    def __init__(self):
        super(Video_Audio_Con_10, self).__init__()

        self.upsample_a1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.conv_last = nn.Conv2d(512, 1, 1, padding=0)

    def forward(self, x, y):
        mix = torch.cat((x, y), 1)

        return mix


class Video_Audio_Con_audio(nn.Module):
    def __init__(self):
        super(Video_Audio_Con_audio, self).__init__()

        self.upsample_a1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.conv_last = nn.Conv2d(512, 1, 1, padding=0)

    def forward(self, x, y):
        mix = torch.cat((x, y), 1)

        return mix
