import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import math

from .backbones import get_backbone

class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=256, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU(inplace=True))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
    
class ZeroMean_Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, bias=True):
        super(ZeroMean_Classifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.zeros(num_classes, in_channels, 1, 1), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        '''if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)'''

    def forward(self, x):
        '''
            x: [BxCxhxw]
        '''
        # weight = self.weight - self.weight.mean(dim=1, keepdim=True)
        out = F.conv2d(x, self.weight, self.bias, stride=1, padding=0, dilation=1, groups=1)
        return out
    
class BiasLayer(nn.Module):
    def __init__(self, num_classes):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
    def forward(self, x):
        return self.alpha * x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, n_base, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=0, **kwargs):
        super(GFSS_Model, self).__init__()
        d_model = 512
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.decoder = PSPModule(2048, out_features=d_model, norm_layer=norm_layer)
        self.classifier = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        )
        
        if is_ft:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=False)
            self.novel_emb = nn.Parameter(torch.zeros(n_novel, d_model), requires_grad=True)
            self.classifier_n = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                ZeroMean_Classifier(d_model, 1, bias=False)
            )
            self.bias_linear = BiasLayer(n_novel+1)
            nn.init.orthogonal_(self.novel_emb)
            self.ft_freeze()
        else:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=True)
            nn.init.orthogonal_(self.base_emb)
            self.novel_emb = None
        self.n_novel = n_novel
        self.use_base = use_base
        self.is_ft = is_ft
        self.criterion = criterion
        self.n_base = n_base

    def init_cls_n(self):
        for param_q, param_k in zip(self.classifier.parameters(), self.classifier_n.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    def train_mode(self):
        self.train()
        # to prevent BN from learning data statistics with exponential averaging
        self.backbone.eval()
        self.decoder.eval()
        # self.classifier.eval()

    def ft_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    @torch.cuda.amp.autocast(enabled=False)
    def orthogonal_decompose(self, feats, bases_b, bases_n=None):
        '''
            feats: [BxCxN]
            bases_b: [1xKxC]
            bases_n: [1xKxC]
            ---
            out_fg:   [BxKxCxN]
            out_bg:   [Bx1xCxN]
        '''
        q = feats.to(torch.float) # [BxCxN]
        s1 = F.normalize(bases_b.to(torch.float), p=2, dim=-1) # [1xKxC]
        # q = feats # [BxCxN]
        # s1 = F.normalize(bases_b, p=2, dim=-1) # [1xKxC]

        proj1 = torch.matmul(s1, q) # [BxKxN]
        out_fg_b = proj1.unsqueeze(2) * s1.unsqueeze(-1) # [BxKxCxN]
        out_bg = q - out_fg_b.sum(1) # [BxCxN]
        if bases_n is not None:
            s2 = F.normalize(bases_n, p=2, dim=-1) # [1xKxC]
            proj2 = torch.matmul(s2, q) # [BxKxN]
            out_fg_n = proj2.unsqueeze(2) * s2.unsqueeze(-1) # [BxKxCxN]
            out_bg = out_bg - out_fg_n.sum(1)# [BxCxN]
            return out_fg_b, out_fg_n, out_bg.unsqueeze(1)
        else:
            out_fg = out_fg_b
            return out_fg, out_bg.unsqueeze(1)

    def forward(self, img, mask=None, img_b=None, mask_b=None, name_n=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            # print('training:', self.training)
            if self.training:
                return self.forward_novel(img, mask, name_n)
            else:
                return self.forward_all(img, mask)
        else:
            return self.forward_base(img, mask)

    def forward_all(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        features = self.backbone.get_image_features(img)
        features = self.decoder(features) # [BxCxhxw]
        B, C, h, w = features.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        out_fg_b, out_fg_n, feats_bg = self.orthogonal_decompose(features.flatten(2), base_emb, novel_emb)

        feats_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = feats_b.contiguous().view(B*(self.n_base+1), C, h, w) # [(BxKb)xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, self.n_base+1, h, w) # [BxKbxhxw]
        # preds1 = self.bias_linear(preds1)

        # feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = out_fg_n.contiguous().view(B*self.n_novel, C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        preds2_n = self.bias_linear(preds2)

        preds = torch.cat([preds1, preds2_n], dim=1)
        # preds = torch.cat([preds2[:,0].unsqueeze(1), preds1, preds2[:,1:]], dim=1)
        # preds = self.bias_linear(preds)
        return preds

    def forward_base(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # x4, x3, _  = self.backbone.base_forward(img, return_list=True)
        features = self.backbone.get_image_features(img)
        features = self.decoder(features) # [BxCxhxw]

        B, C, h, w = features.shape
        cls_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]

        n_class = 1 + cls_emb.shape[1]
        features = features.flatten(2) # [BxCxN]
        feats_fg, feats_bg = self.orthogonal_decompose(features, cls_emb)

        feats_all = torch.cat([feats_bg, feats_fg], dim=1) # [Bx(1+K)xCxN]
        feats_all = feats_all.contiguous().view(B*n_class, C, h, w) # [(Bx(1+K))xCxhxw]

        preds = self.classifier(feats_all) # [(Bx(1+K))x1xhxw]
        preds = preds.view(B, n_class, h, w) # [Bx(1+K)xhxw]

        if self.criterion is not None and mask is not None:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1).squeeze(0) # [KbasexC]
            proto_sim = torch.matmul(cls_emb, cls_emb.t()) # [KbasexKbase]
            return self.criterion(preds, mask, proto_sim=proto_sim)
        else:
            return preds

    def forward_novel(self, img, mask):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # with torch.no_grad():
        # img_full = torch.cat([img, img_b], dim=0)
        features_full = self.backbone.get_image_features(img)
        features_full = self.decoder(features_full) # [BxCxhxw]

        B, C, h, w = features_full.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        n_class = 1 + base_emb.shape[1] + novel_emb.shape[1]
        features_full = features_full.flatten(2) # [BxCxN]
        out_fg_b, out_fg_n, feats_bg = self.orthogonal_decompose(features_full, base_emb, novel_emb)

        feats_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = feats_b.reshape(B*(1+self.n_base), C, h, w) # [(Bx(1+Kb))xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, 1+self.n_base, h, w) # [BxKbxhxw]
        # preds1 = self.bias_linear(preds1)

        # feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = out_fg_n.reshape(B*self.n_novel, C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        preds2_n = self.bias_linear(preds2)
        
        preds = torch.cat([preds1, preds2_n], dim=1)

        # relabeling
        '''mask_new = []
        for b in range(B//2):
            bg_mask = mask_b[b] == 0
            bg_out = preds2[B//2+b] # [(1+Kn)xhxw]
            bg_out = F.interpolate(input=bg_out.unsqueeze(0), size=mask_b[b].shape, mode='bilinear', align_corners=True)

            bg_idx = torch.argmax(bg_out.squeeze(0), dim=0) # [hxw]
            bg_idx[bg_idx>0] += self.n_base
            mask_b[b][bg_mask] = bg_idx[bg_mask]
            mask_new.append(mask_b[b])
        mask_new = torch.stack(mask_new, dim=0)'''

        if self.criterion is not None and mask is not None:
            with torch.cuda.amp.autocast(enabled=False):
                # mask_all = torch.cat([mask, mask_new], dim=0)
                novel_emb = F.normalize(novel_emb.to(torch.float), p=2, dim=-1) # [1xKnovelxC]
                novel_emb = novel_emb.reshape(-1, C) # [KnxC]
                all_emb = torch.cat([novel_emb, F.normalize(base_emb.to(torch.float).squeeze(0), p=2, dim=-1)], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]

                return self.criterion(preds.to(torch.float), mask, is_ft=True, proto_sim=proto_sim)
        else:
            return preds
