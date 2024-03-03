import torch
import torch.nn as nn
from .models.lseg_net import LSegNet
from typing import List, Tuple, Optional, Union

class LSegModule(nn.Module):
    def __init__(self, backbone, block_depth, activation, resolution=None):
        super().__init__()

        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        # print('** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        self.net = LSegNet(
            backbone=backbone,
            block_depth=block_depth,
            activation=activation,
        )

        self.mean = norm_mean
        self.std = norm_std
        
        self.img_resolution=resolution

    def evaluate_random(self, x, labelset):
        pred = self.net.forward(x, labelset)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return pred
    
    def forward_image_only(self, x, flip=False):
        return self.net.forward_image_only(x, flip)
    
    def forward_text_only(self, labelset='', device='cuda'):
        return self.net.forward_text_only(labelset, device)
    
    def forward_logits(self, image_features, text_features, flip=False):
        return self.net.forward_logits(image_features, text_features, flip)
    
    def forward(self, x, text_features=None, labelset: Optional[Union[List[str], Tuple]]=None, flip=False):
        assert not (text_features is None and labelset is None)
        if text_features is None:
            text_features = self.forward_text_only(labelset)
        if flip:
            image_features, image_features_flip = self.forward_image_only(x), self.forward_image_only(x, True)
            out = self.forward_logits(image_features, text_features)+self.forward_logits(image_features_flip, text_features, True)
            return out
        else:
            image_features = self.forward_image_only(x)
            out = self.forward_logits(image_features, text_features)
            return out        
    
    def set_imshape(self, x=None):
        if self.img_resolution is None:
            self.net.set_imshape(x)
        elif self.img_resolution == 256:
            self.net.imshape = (1, 512, 128, 128)
        elif self.img_resolution == 1024:
            self.net.imshape = (1, 512, 512, 512)
            pass