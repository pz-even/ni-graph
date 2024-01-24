import os.path as osp
import numpy as np
import glob
import scipy.io as sio
import torch
from torch_geometric.data import Dataset, Data

class MakeGraph(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MakeGraph, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        files = glob.glob(osp.join(self.raw_dir, '*.mat'))
        file_ = [f.split('/')[4] for f in files]
        return file_
    
    @property
    def processed_file_names(self):
        files = glob.glob(osp.join(self.raw_dir, '*.mat'))
        file_ = [f.split('/')[4] for f in files]
        file_ = [f.replace('.mat','.pt') for f in file_]
        return file_

    def len(self):
        return len(self.processed_file_names)
    
    def download(self):
        if all([osp.exists(f) for f in self.raw_paths]): return
        print('NO DATA!')

    def process(self):
        for raw_path in self.raw_paths:
            event = sio.loadmat(raw_path)
            x = torch.tensor(event['x'], dtype=torch.float) 
            edge_index = torch.tensor(np.array(event['edge_index'], np.int16), dtype=torch.long) 
            edge_attr = torch.tensor(event['edge_attr'], dtype=torch.float)
            y = torch.tensor(event['y'], dtype=torch.long)
            data = Data(x=x,
                        edge_index=edge_index.t().contiguous(),
                        edge_attr=edge_attr, 
                        y=y.squeeze(0))

            if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            file_name = raw_path.split('/')[4].replace('.mat','.pt')
            torch.save(data, osp.join(self.processed_dir, file_name))

    def get(self, idx):
        data = torch.load(osp.join(self.processed_paths[idx]))
        return data