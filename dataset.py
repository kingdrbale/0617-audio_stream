import torch.utils.data as data
from audio_process import *
from mean_std import *


class LoadDataset(data.Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.specs = []
        self.gt = []

        specs_path = self.root_folder.split('frames')[0] + 'specs' + self.root_folder.split('frames')[1]
        print("Specs Root : ", specs_path, "\n")

        self.specs = [os.path.join(specs_path, f) for f in os.listdir(specs_path) if f.endswith('.jpg')]
        self.gt = [os.path.join(self.root_folder, f) for f in os.listdir(self.root_folder) if f.endswith('.jpg')]

        self.specs.sort()
        self.gt.sort()

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, item):
        sp_name = self.specs[item]
        gt_name = self.gt[item]

        audio_data, gt_data = load_single_sp_gt(sp_name, gt_name)

        return audio_data, gt_data


class LoadTestData:
    def __init__(self, test_root):
        # print(test_root, specs_root)
        self.frames = [test_root + f for f in os.listdir(test_root) if f.endswith('.jpg')]
        self.frames = sorted(self.frames)
        self.size = len(self.frames)
        self.index = 0

    def load_data(self):
        frame = self.data_loader_frames(self.frames[self.index])
        name = self.frames[self.index].split('/')[-1]
        self.index += 1

        return frame, name, self.index

    def data_loader_frames(self, name):
        audio = Image.open(name).convert('RGB')
        # convert a PIL image to tensor (HWC) in range [0,255] to a torch.Tensor(CHW) in the range [0.0,1.0]
        audio_data = F.to_tensor(audio)

        return audio_data
