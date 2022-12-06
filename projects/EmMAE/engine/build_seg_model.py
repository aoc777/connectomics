import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .mae import *
from connectomics.model.utils.misc import IntermediateLayerGetter
from timm.models.vision_transformer import Block

backbone_dict = {
    'vit-base': mae_vit_base_patch16_dec512d8b,
    'vit-large': mae_vit_large_patch16_dec512d8b,
    'vit-huge': mae_vit_huge_patch14_dec512d8b,}

backbone_param_dict = {
    'vit-base': [12,768],
    'vit-large': [24,1152],
    'vit-huge': [32,1152],}

def build_backbone(backbone_type: str,
                   load: bool,
                   **kwargs):
    assert backbone_type in ['vit-base', 'vit-large', 'vit-huge']
    feat_keys = [None]*2
    return_layers = {'patch_embed': feat_keys[0],
                     'blocks': feat_keys[1],}

    backbone = backbone_dict[backbone_type]()

    if load:
        print('loading pre-trained model')
        checkpoint = torch.load('/ygs/personal/chengao/connectomics_project/mae_model/test/checkpoint_400000.pth.tar',
                                map_location='cpu') #path to pre-trained model
        backbone.load_state_dict(checkpoint['state_dict'])
    return IntermediateLayerGetter(backbone, return_layers)

class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size[0] * patch_size[1] * patch_size[2]))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=(patch_size[0] , patch_size[1] , patch_size[2]),
                                          stride=(patch_size[0] , patch_size[1] , patch_size[2]))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        return x

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class UNETR(nn.Module):
    def __init__(self, img_shape=(128, 128, 128), input_dim=1, output_dim=3,
                 dropout=0.1, patch_size=16, pretrain = True, vit_backbone='vit-base',
                 num_heads=16, cube_size = [6,96,96], mlp_ratio=4.):
        super().__init__()
        self.output_dim = output_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        # self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]
        self.patch_dim = [6,6,6]
        # self.embed = Embeddings(in_chans, embed_dim, cube_size, [1,16,16], dropout)
        # Transformer Encoder
        self.backbone = build_backbone(vit_backbone,pretrain)
        depth,self.embed_dim = backbone_param_dict[vit_backbone]
        self.depth = depth//4
        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(self.embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(self.embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )

    def forward(self, x):
        embed = self.backbone.patch_embed(x)
        feat = []
        z = embed
        for layers in self.backbone.blocks:
            z = layers(z)
            feat.insert(0,z)
        ext_feat = []

        for i in range(1,5):
            ext_feat.insert(0, feat[i * self.depth - 1])
        z0, z3, z6, z9, z12 = x, *ext_feat

        # z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
