import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from dataloaders.eeg_transforms import ToTensor


class BCICompet2aIV(Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        self.args = args
        
        import warnings
        warnings.filterwarnings('ignore')
        
        toTenor = ToTensor()
        
        self.data, self.label = self.get_brain_data()
        
        self.data = toTenor(self.data)
        # self.label = toTenor(self.label)
        
        print(self.data.shape, self.label.shape)
        

        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    
    def get_brain_data(self):
        filelist = sorted(glob(f'datasets/{self.args.train_mode}/*X.npy'))
        label_filelist = sorted(glob(f'datasets/{self.args.train_mode}/*Y.npy'))
        
        
        if self.args.train_mode == 'train':
            filelist_arr, label_filelist_arr = [], []
            
            for file, label in zip(filelist, label_filelist):
                filelist_arr.append(np.load(file))
                label_filelist_arr.append(np.load(label))
            
            # import pdb; pdb.set_trace()
            filelist = np.concatenate((filelist_arr))
            label_filelist = np.concatenate((label_filelist_arr))
        
        elif self.args.train_mode == 'validation':
            filelist = np.load(filelist[0])
            label_filelist = np.load(label_filelist[0])
            
        label_filelist = torch.tensor(label_filelist).long()
        
        return filelist, label_filelist
        

class BCICompet2aIV_TEST(Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        self.args = args
        
        import warnings
        warnings.filterwarnings('ignore')
        
        toTenor = ToTensor()
        
        self.data = self.get_brain_data()
        
        self.data = toTenor(self.data)
        
        print(self.data.shape)
        

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
    def get_brain_data(self):
        filelist = sorted(glob(f'datasets/{self.args.train_mode}/*X.npy'))
        
        
        filelist = np.load(filelist[0])
            
        return filelist
        
    