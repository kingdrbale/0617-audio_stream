from datetime import datetime
from torch import nn
from mean_std import *
now = str(datetime.now())

epochs = 25
batch_size = 20
lr = 1e-3
decay = 5
save_hop = 2
obj_func = nn.BCEWithLogitsLoss()
fixed = 0
f_num = 7
specs_stack = 7
'''
MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]
'''
MEAN = train_specs_mean
STD = train_specs_std

train_data_path = './Data/OP_Train_DATA/frames/'
test_data_path = './Data/OP_Test_DATA/specs/'
test_data_path1 = './Data/DAVSOD_Test_DATA/frames_and_GT_part/easy/'
# checkpoint_load = './pre-trained/model.pth.tar'
checkpoint_load = './pre-trained/model_cpd_400615_Adam_0.0001_20_BCEWithLogitsLoss_8.pth'
# checkpoint_load = './pre-trained/model_cpd_500608_Adam_1.0000000000000002e-07_20_BCEWithLogitsLoss_40.pth'
checkpoint_name = 'cpd_' + str(epochs) + now[5:7] + now[8:10]
checkpoint_save = './checkpoint/'
