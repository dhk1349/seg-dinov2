import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import timm


class DINO2SEG(nn.Module):
    def __init__(self, num_cls):
        super(DINO2SEG, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_class = num_cls 
        # for param in self.backbone.parameters():
        #     param.requires_grad=True

        switch = False
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                if 'blocks.4.' in name:
                    switch = True
                if switch:
                    param.requires_grad=True
                else:
                    param.requires_grad=False



        # token projection 
        # linear
        self.l1 = nn.Linear(768, 512)
        # self.l3 = nn.Linear(num_cls, num_cls)
        self.relu = nn.ReLU()
        # self.out = nn.Conv2d(512, num_cls, 1)
        self.out = nn.Conv2d(512, num_cls, 3, 1, padding='same', padding_mode='replicate')
        # upsample
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample((448, 448))
    
    def forward(self, x):
        bs = x.shape[0]
        # x = torch.permute(x, (0, 3, 1, 2))
        out = self.backbone.forward_features(x.float()) 
        
        # out = self.mlp_head(out["x_norm_patchtokens"]) # dinov2
        out = out["x_norm_patchtokens"]
        out = self.l1(out)
        out = torch.permute(out, (0, 2, 1))
        out = out.view((bs, 512, 32, 32))
        # out = self.up1(out)
        # out = self.up2(out)
        out = self.up3(out)
        # out = out.resize((bs, 128, 518, 518))
        out = self.out(out)

        return out


    def save(self, path):
        # save mlp_head and not backbone
        torch.save({
            "mlp_head": self.mlp_head.state_dict(),
            }, path)

    def load(self, weight):
        self.mlp_head.load_state_dict(weight)
