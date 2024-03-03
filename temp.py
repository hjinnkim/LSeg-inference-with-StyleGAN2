import torch
import torch.nn as nn

class temp(nn.Module):
    def __init__(self):
        super(temp, self).__init__()
    
    def forward(self, i):
        return i+1

class list_module(nn.Module):
    def __init__(self):
        super(list_module, self).__init__()
        self.attention = {}
        
        self._list = {}
        for i in range(4):
            self._list[i]=temp()
            self._list[i].attention = self.attention
        
    def forward(self):
        output = 0
        for key, val in self._list.items():
            output = val(output)
            val.attention[key]=output
            print(key, output)
        print('=============')
        print(self.attention)

a = list_module()
a()