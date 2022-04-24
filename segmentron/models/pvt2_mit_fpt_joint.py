import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
import math
from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import  _FCNHead
from ..config import cfg
from ..data.dataloader import datasets
from .backbones.pvtv2_mix_transformer import Attention, Mlp


__all__ = ['PVTV2_FPT_JOINT']



@MODEL_REGISTRY.register(name='PVTV2_FPT_JOINT')
class PVTV2_FPT_JOINT(SegBaseModel):
    """ PVTv2 with 2 joint FPT decoder. """

    def __init__(self):
        super().__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        elif self.backbone.startswith('resnet18'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('pvt'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('resnet34'):
            c1_channels = 64
            c4_channels = 512
        elif self.backbone.startswith('hrnet_w18_small_v1'):
            c1_channels = 16
            c4_channels = 128
        else:
            c1_channels = 256
            c4_channels = 2048

        vit_params = cfg.MODEL.TRANS4TRANS
        hid_dim = cfg.MODEL.TRANS4TRANS.hid_dim

        assert cfg.TRAIN.CROP_SIZE[0] == cfg.TRAIN.CROP_SIZE[1]\
               == cfg.TRAIN.BASE_SIZE == cfg.TEST.CROP_SIZE[0] == cfg.TEST.CROP_SIZE[1]
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 32) ** 2

        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['emb_chans'] = cfg.MODEL.EMB_CHANNELS

        vit_params['nclass'] = self.nclass
        self.fpt_head_1 = FPTHead(vit_params, c1_channels=c1_channels, c4_channels=c4_channels, hid_dim=hid_dim)
        vit_params['nclass'] = datasets[cfg.DATASET2.NAME].NUM_CLASS
        self.fpt_head_2 = FPTHead(vit_params, c1_channels=c1_channels, c4_channels=c4_channels, hid_dim=hid_dim)
        decoders = ['fpt_head_1', 'fpt_head_2']
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
            decoders.append('auxlayer')
        self.__setattr__('decoder', decoders)


    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        outputs = list()
        x_1 = self.fpt_head_1(c1, c2, c3, c4)
        x_1 = F.interpolate(x_1, size, mode='bilinear', align_corners=True)
        outputs.append(x_1)

        x_2 = self.fpt_head_2(c1, c2, c3, c4)
        x_2 = F.interpolate(x_2, size, mode='bilinear', align_corners=True)
        outputs.append(x_2)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

class ProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.proj = nn.Linear(in_chans, emb_chans)
        self.norm = nn.LayerNorm(emb_chans)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.proj(x.flatten(2).transpose(1, 2))
        # x = self.act1(self.bn1(self.fc1(x))).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class HeadBlock(nn.Module):
    def __init__(self, in_chans=512, emb_chans=64, num_heads=2, sr_ratio=4):
        super().__init__()
        self.proj = ProjEmbed(in_chans=in_chans, emb_chans=emb_chans)
        self.norm1 = partial(nn.LayerNorm, eps=1e-6)(emb_chans)
        self.attn = Attention(emb_chans, num_heads=num_heads, sr_ratio=sr_ratio)
        self.drop_path = nn.Identity()
        self.norm2 = partial(nn.LayerNorm, eps=1e-6)(emb_chans)
        mlp_ratio = 2
        mlp_hidden_dim = int(emb_chans * mlp_ratio)
        self.mlp = Mlp(in_features=emb_chans, hidden_features=mlp_hidden_dim, act_layer=nn.Hardswish)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x



class FPTHead(nn.Module):
    def __init__(self, vit_params, c1_channels=256, c4_channels=2048, hid_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        sr_ratio = [4, 4, 4, 1]
        emb_chans = vit_params['emb_chans'] 
        self.head1 = HeadBlock(in_chans=64, emb_chans=emb_chans, sr_ratio=sr_ratio[0])
        self.head2 = HeadBlock(in_chans=128, emb_chans=emb_chans, sr_ratio=sr_ratio[1])
        self.head3 = HeadBlock(in_chans=320, emb_chans=emb_chans, sr_ratio=sr_ratio[2])
        self.head4 = HeadBlock(in_chans=512, emb_chans=emb_chans, sr_ratio=sr_ratio[3])

        self.pred = nn.Conv2d(emb_chans, vit_params['nclass'], 1)


    def forward(self, c1, c2, c3, c4):
        size = c1.size()[2:]

        c4 = self.head4(c4) 
        out = F.interpolate(c4, size, mode='bilinear', align_corners=True)

        c3 = self.head3(c3)
        out += F.interpolate(c3, size, mode='bilinear', align_corners=True)

        c2 = self.head2(c2)
        out += F.interpolate(c2, size, mode='bilinear', align_corners=True)

        out += self.head1(c1)
        out = self.pred(out)
        return out

