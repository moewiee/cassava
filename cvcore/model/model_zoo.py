import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.nn.functional import dropout
import random


class NormSoftmax(nn.Module):
    def __init__(self, in_features, out_features, temperature=1.):
        super(NormSoftmax, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data)

        self.ln = nn.LayerNorm(in_features, elementwise_affine=False)
        self.temperature = nn.Parameter(torch.Tensor([temperature]))

    def forward(self, x):
        x = self.ln(x)
        x = torch.matmul(F.normalize(x), F.normalize(self.weight))
        x = x / self.temperature
        return x


class EfficientNet(nn.Module):
    """
    EfficientNet B0-B8.
    Args:
        cfg (CfgNode): configs
    """
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()
        self.cfg = cfg

        backbone = timm.create_model(
            model_name=self.cfg.MODEL.NAME,
            pretrained=self.cfg.MODEL.PRETRAINED,
            in_chans=self.cfg.DATA.INP_CHANNEL,
            drop_path_rate=self.cfg.MODEL.DROPPATH,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        ### Original blocks ###
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        if self.cfg.MODEL.POOL == "adaptive_pooling":
            self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.num_features = backbone.num_features
        if self.cfg.MODEL.HYPER:
            self.num_features = backbone.num_features + self.block4[-1].bn3.num_features + \
                                self.block5[-1].bn3.num_features
        ### Baseline head ###
        if self.cfg.MODEL.CLS_HEAD == "linear":
            self.fc = nn.Linear(self.num_features, self.cfg.MODEL.NUM_CLASSES)
        del backbone

    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b5 = x
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return b4,b5,x

    def forward(self, x):
        with autocast():
            b4, b5, x = self._features(x)
            x = self.global_pool(x)
            if self.cfg.MODEL.HYPER:
                b4 = self.global_pool(b4)
                b5 = self.global_pool(b5)
                x = torch.cat([x, b4, b5], 1)
            x = torch.flatten(x, 1)
            if self.cfg.MODEL.DROPOUT > 0.:
                x = torch.nn.functional.dropout(x, self.cfg.MODEL.DROPOUT, training=self.training)
            logits = self.fc(x)
            return logits

def build_model(cfg):
    model = None
    if "efficientnet" in cfg.MODEL.NAME:
        model = EfficientNet
    return model(cfg)
