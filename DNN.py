import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

class DNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output = out_features
        
        self.gru = nn.GRU(in_features, hidden_dim, 1, batch_first=True, dropout=0.1) #num layer=5 / 3 / 1
        
        #self.fc1 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2) 
        #self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        #self.fc4 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc5 = nn.Linear(hidden_dim*2, hidden_dim) 
        self.out = nn.Linear(hidden_dim, out_features)

    def forward(self, x):        
        x = x.transpose(1,2)
        x, hidden = self.gru(x)
        
        x = x[:, -1, :]
        #x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)
        x = self.fc5(x)
        x = self.out(x)
        return x
  


class MyDataset(Dataset):
    def __init__(self, data, window, offset, target_col):
        self.data = torch.Tensor(data)
        self.window = window #num_points_for_train
        self.used_cols = [x for x in range(data.shape[1]) if x!=target_col] # X = except target column
        self.target_col = target_col #y
        self.offset = offset #predict offset
        
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index:index+self.window, self.used_cols].T #for train window length
        y = self.data[index+self.window+self.offset, self.target_col] #after offset = 30m
        return x, y

    def __len__(self):
        return len(self.data) -  self.window - self.offset # train data length
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())