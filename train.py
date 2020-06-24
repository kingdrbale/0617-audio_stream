import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import LoadDataset
from hyper import *
from model.CPD_models import CPD_3DResNet

torch.cuda.set_device(0)


class Train(object):
    def __init__(self):
        super(Train, self).__init__()
        self.train_list = [os.path.join(train_data_path, p) for p in os.listdir(train_data_path)]
        self.model = CPD_3DResNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer_name = str(self.optimizer).split(' ')[0]
        self.func_name = str(obj_func).split('()')[0]
        self.model = self.model.cuda()
        self.loss_dict = []
        self.lr = lr
        # self.model.load_state_dict(torch.load(checkpoint_load))

        save_model = torch.load(checkpoint_load)
        # save_model = torch.load(checkpoint_load)['state_dict']
        now_model = self.model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in now_model.keys()}
        now_model.update(state_dict)
        self.model.load_state_dict(now_model)
        '''
        dict_name = list(state_dict)
        for i1, p1 in enumerate(dict_name):
            print(i1, p1)

        exit(0)
        
        f = 0
        # 329
        for name, para in self.model.named_parameters():
            f += 1
        self.flag = [1 for x in range(0, f)]

        ff = 0
        for name, para in self.model.named_parameters():
            for i1, p1 in enumerate(dict_name):
                if name == p1:
                    ff = i1
                    self.flag[ff] = 0
                    # print(i1, name)
        # exit(0)
        '''

        print("\n================\nPre-Model setup OK !\n================")
        # self.checkpoint_save = 'No_Pretrained'
        torch.cuda.empty_cache()
        self.checkpoint_save = checkpoint_save + 'cpd_' + now[5:7] + now[8:10]

    def train(self):
        # 新建保存 checkpoint 的文件夹
        if not os.path.exists(self.checkpoint_save):
            os.makedirs(self.checkpoint_save)

        for epoch in range(epochs):
            print("=====This is Epoch =====", epoch)
            '''
            if epoch < 2:
                for idx, p in enumerate(self.model.parameters()):
                    if self.flag[idx] == 0:
                        p.requires_grad = False
            else:
                for idx, p in enumerate(self.model.parameters()):
                    p.requires_grad = True
            
            fixed = 1
            '''

            if epoch % decay == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * 0.1
            self.lr = param_group['lr']
            print("learning rate : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
            learning_rate = str(self.optimizer.state_dict()['param_groups'][0]['lr'])

            dir_num = 0

            for dir in self.train_list:
                i = 0
                train_set = LoadDataset(dir)
                train_loader = DataLoader(train_set, batch_size=batch_size,
                                          num_workers=0, shuffle=True)

                for idx, train_data in enumerate(train_loader):
                    audio, gt = train_data
                    # print('Train : ', audio.shape, gt.shape)
                    # exit(0)
                    # torch.Size([bs, 3, f_num, 256, 320])
                    # torch.Size([bs, 3, f_num, 256, 320])
                    # torch.Size([bs, 1, 256, 320])

                    # image = Variable(image).cuda()
                    audio = Variable(audio).cuda()
                    gt = Variable(gt).cuda()

                    det = self.model(audio)
                    # print('Result : ', det.shape, gt.shape)
                    # exit(0)

                    # loss1 = obj_func(att, gt)
                    loss2 = obj_func(det, gt)
                    loss = loss2

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_dict.append(loss.item())

                    i += 1

                print('Epoch: [{}], Root_folder: {}, Frames: {}, Enumerate: {}\n=====Loss: {:.6f}=====\n'.format(
                    epoch, dir, len(train_set), i, loss.item()))
                '''
                dir_num += 1
                if dir_num == 500:
                    break
                '''
            if epoch % save_hop == 0:
                torch.save(self.model.state_dict(),
                       os.path.join(self.checkpoint_save, 'model_%s_%s_%s_%d_%s_%d.pth' % (checkpoint_name,
                                                                                        self.optimizer_name,
                                                                                        learning_rate,
                                                                                        batch_size,
                                                                                        self.func_name,
                                                                                        epoch)))

        end_time = str(datetime.now())[11:13] + str(datetime.now())[14:16]
        save_name = './charts/' + self.optimizer_name + '_' + str(lr) + '_' + self.func_name + '_' + end_time + '.png'
        plt.title(self.optimizer_name + '_' + str(lr) + '_' + str(epochs) + '_' + str(decay) + '_' + str(fixed) + end_time)
        plt.xlabel('batch')
        plt.ylabel('loss')

        plt.plot(self.loss_dict)
        plt.savefig(save_name)
        plt.show()


if __name__ == '__main__':
    t = Train()
    t.train()
