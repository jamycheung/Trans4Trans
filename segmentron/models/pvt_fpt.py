import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _FCNHead
from ..config import cfg
from .backbones.pvt import Attention, Mlp


__all__ = ['PVT_FPT']


@MODEL_REGISTRY.register(name='PVT_FPT')
class PVT_FPT(SegBaseModel):

    def __init__(self, ncls=None):
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

        assert cfg.AUG.CROP == False and cfg.TRAIN.CROP_SIZE[0] == cfg.TRAIN.CROP_SIZE[1]\
               == cfg.TRAIN.BASE_SIZE == cfg.TEST.CROP_SIZE[0] == cfg.TEST.CROP_SIZE[1]
        c4_HxW = (cfg.TRAIN.BASE_SIZE // 32) ** 2

        vit_params['decoder_feat_HxW'] = c4_HxW
        vit_params['nclass'] = self.nclass if ncls is None else ncls
        vit_params['emb_chans'] = cfg.MODEL.EMB_CHANNELS

        self.fpt_head = FPTHead(vit_params)
        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['fpt_head', 'auxlayer'] if self.aux else ['fpt_head'])


    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)

        outputs = list()
        x = self.fpt_head(c1, c2, c3, c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
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

    def forward(self, x):
        x = self.proj(x.flatten(2).transpose(1, 2))
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
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x



class FPTHead(nn.Module):
    def __init__(self, vit_params):
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

