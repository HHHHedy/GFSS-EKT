import torch
import torch.nn as nn
from torch.nn import functional as F
import random

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
        self.alpha = nn.Parameter(torch.ones(num_classes), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(), requires_grad=False)
    def forward(self, x):
        # bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        alpha = self.alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # print(x.shape, bias.shape, self.alpha.shape)
        return alpha * x
    

class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, n_base, fold, shot=1, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=0, **kwargs):
        super(GFSS_Model, self).__init__()
        self.shot = shot
        num_classes = n_base + n_novel  # 20
        if fold == -1:
            # training with all classes
            base_classes = set(range(1, num_classes+1))
            novel_classes = set()
            
        else:
            interval = num_classes // 4
            # base classes = all classes - novel classes
            base_classes = set(range(1, num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
            # novel classes
            novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))
        self.base_classes = [x for x in base_classes]
        self.novel_classes = [x for x in novel_classes]
        d_model = 512
        self.backbone = get_backbone(norm_layer=norm_layer, **kwargs)
        self.decoder = PSPModule(2048, out_features=d_model, norm_layer=norm_layer)
        # self.combine = nn.Conv2d(2 * d_model, d_model, kernel_size=1, bias=False)
        self.classifier = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 1, kernel_size=1, bias=False)
        )
        self.n_novel = n_novel
        self.use_base = use_base
        self.is_ft = is_ft
        self.criterion = criterion
        self.n_base = n_base
        if is_ft:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=False)
            self.novel_emb = nn.Parameter(torch.zeros(n_novel, d_model), requires_grad=True)
            self.k_linear_b = nn.Linear(d_model, d_model)
            self.q_linear_n = nn.Linear(d_model, d_model)
            self.v_linear_b = nn.Linear(d_model, d_model)
            self.fuse_linear_n = nn.Linear(2*d_model, d_model)
            # self.cls_emb = nn.Parameter(torch.zeros(num_classes, d_model), requires_grad=True)
            # self.base_emb = nn.ParameterList([param for param in self.cls_emb[:n_base]])
            # self.novel_emb = nn.ParameterList([param for param in self.cls_emb[n_base:]])
            # self.decoder_n = PSPModule(2048, out_features=d_model, norm_layer=norm_layer)
            self.classifier_n = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                ZeroMean_Classifier(d_model, 1, bias=False)
            )
            self.bias_linear = BiasLayer(1)
            nn.init.orthogonal_(self.novel_emb)  # initialize a set of orthogonal vectors
            self.ft_freeze()
            # self.mae_loss = nn.L1Loss(reduction='mean')
            # self.mse_loss = nn.MSELoss(reduction='mean')
            self.seg_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        else:
            self.base_emb = nn.Parameter(torch.zeros(n_base, d_model), requires_grad=True)
            nn.init.orthogonal_(self.base_emb)
            self.novel_emb = None
        

    def init_cls_n(self):
        for param_q, param_k in zip(self.classifier.parameters(), self.classifier_n.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
        # for param_b, param_n in zip(self.decoder.parameters(), self.decoder_n.parameters()):
            # param_n.data.copy_(param_b.data)  # initialize
        

    def train_mode(self):
        self.train()
        # to prevent BN from learning data statistics with exponential averaging
        self.backbone.eval()
        self.decoder.eval()
        # self.decoder_n.eval()
        # self.classifier.eval()

    def ft_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        # for param in self.base_emb:
            # param.requires_grad = False
        # for param in self.classifier_n.parameters():
            # param.requires_grad = False
        # for param in self.bias_linear.parameters():
            # param.requires_grad = False
    
    @torch.cuda.amp.autocast(enabled=False)
    def orthogonal_decompose(self, feats, bases_b=None, bases_n=None):
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
        # s1 = s1.repeat(q.size(0), 1, 1)  # [BxKxC]
        # q = feats # [BxCxN]
        # s1 = F.normalize(bases_b, p=2, dim=-1) # [1xKxC]

        proj1 = torch.matmul(s1, q) # [BxKxN] similarity of q to s1
        out_fg_b = proj1.unsqueeze(2) * s1.unsqueeze(-1) # [BxKxCxN] projection of q on s1
        out_bg = q - out_fg_b.sum(1) # [BxCxN]
        
        if bases_n is not None:
            s2 = F.normalize(bases_n, p=2, dim=-1) # [1xKxC]
            proj2 = torch.matmul(s2, q) # [BxKxN]
            out_fg_n = proj2.unsqueeze(2) * s2.unsqueeze(-1) # [BxKxCxN]
            pure_out_bg = out_bg - out_fg_n.sum(1)# [BxCxN]
            # out_bg_norm = F.normalize(out_bg, p=2, dim=-1) # [BxCxN]
            # proj_bg = F.cosine_similarity(out_bg_norm, q, dim=1) # [BxN]
            proj_bg = torch.norm(pure_out_bg, p=2, dim=1) # [BxN]
            proj = torch.cat([proj_bg.unsqueeze(1), proj1, proj2], dim=1) # [Bx(K+1)xN]
            # bg_norm = F.normalize(out_bg, p=2, dim=1) # [BxCxN]
            # unit_bg = torch.mean(bg_norm, dim=-1) # [BxC]
            # units = torch.cat([unit_bg.unsqueeze(1), s1], dim=1) # [Bx(K+1)xC]

            return out_fg_b, out_fg_n, pure_out_bg.unsqueeze(1), proj
        else:
            out_fg = out_fg_b
            proj_bg = torch.norm(out_bg, p=2, dim=1) # [BxN]
            proj = torch.cat([proj_bg.unsqueeze(1), proj1], dim=1) # [Bx(K+1)xN]
            
            return out_fg, out_bg.unsqueeze(1), proj
        '''if bases_n is not None:
            s2 = F.normalize(bases_n, p=2, dim=-1) # [1xKxC]
            proj2 = torch.matmul(s2, q) # [BxKxN]
            out_fg_n = proj2.unsqueeze(2) * s2.unsqueeze(-1) # [BxKxCxN]
            out_bg = out_bg - out_fg_n.sum(1)# [BxCxN]
            return out_fg_b, out_fg_n, out_bg.unsqueeze(1)
        else:
            out_fg = out_fg_b
            return out_fg, out_bg.unsqueeze(1)'''
        
    def Weighted_GAP(self, supp_feat, mask):
        mask = F.interpolate(mask.float().unsqueeze(1), size=supp_feat.size()[-2:], mode='nearest')  # [5, 1, 60, 60]
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2], supp_feat.shape[-1]
        area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def forward(self, img, mask=None, img_b=None, img_s=None, s_img=None, s_mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            if self.training:
                return self.forward_novel(img, mask, img_b, img_s)
            else:
                return self.forward_all(img, mask)
        else:
            return self.forward_base(img, mask)

    def forward_all(self, img, mask=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        features = self.backbone.base_forward(img)
        features = self.decoder(features) # [BxCxhxw]
        # features_n = self.decoder_n(features) # [BxCxhxw]

        B, C, h, w = features.shape

        #base_emb = torch.stack([param for param in self.base_emb])
        #novel_emb = torch.stack([param for param in self.novel_emb])
        base_emb = self.base_emb # [KbasexC]
        novel_emb = self.novel_emb # [KnovelxC]
        '''base_emb_exp = base_emb.unsqueeze(0)
        novel_emb_exp = novel_emb.unsqueeze(1)
        cos_sim = F.cosine_similarity(base_emb_exp, novel_emb_exp, dim=-1)  # [5,15]
        attention_weights = F.softmax(cos_sim, dim=-1)
        attended_novel_emb = torch.matmul(attention_weights, base_emb)
        attended_novel_emb = attended_novel_emb.unsqueeze(0) # [1xKnovelxC]'''
        # base to novel attention
        K_b = self.k_linear_b(base_emb)
        Q_n = self.q_linear_n(novel_emb)
        V_b = self.v_linear_b(base_emb)
        # 步骤1: 计算点积得分
        # torch.matmul 可以用来计算矩阵乘法。这里我们需要将 novel_units 与 base_units 的转置相乘。
        scores_b_n = torch.matmul(Q_n, K_b.transpose(0, 1))  # 形状为 [5, 15]

        # 步骤2: 应用 softmax 正规化
        # 对 scores 的最后一个维度（即对每个 novel unit 对所有 base units 的分数）进行 softmax
        attention_weights = F.softmax(scores_b_n, dim=-1)  # 形状仍为 [5, 15]

        # 步骤3: 计算加权和
        # 使用扩展的 attention weights 乘以 base_units
        attended_novel_emb = torch.matmul(attention_weights, V_b)  # 结果形状为 [5, 512]
        attended_novel_emb = attended_novel_emb.unsqueeze(0)
        # novel to base attention
        

        # novel_emb = torch.cat([novel_emb, attended_novel_emb], dim=1) # [Knx(C+C)]
        # novel_emb = self.fuse_linear_n(novel_emb)
        base_emb = base_emb.unsqueeze(0) # [1xKbasexC]
        novel_emb = novel_emb.unsqueeze(0) # [1xKnovelxC]
        # novel_emb = 0.5 * novel_emb + 0.5 * attended_novel_emb
        # cls_emb = self.cls_emb.unsqueeze(0) # [1x1xC]
        # cls_emb = torch.cat([novel_emb, attended_novel_emb], dim=1) # [1x(Kbase+Knovel)xC]
        # cls_emb = self.cls_emb.unsqueeze(0) 
        n_class = 1 + base_emb.shape[1] + novel_emb.shape[1]

        out_fg_b, out_fg_n, feats_bg, sim_maps = self.orthogonal_decompose(features.flatten(2), bases_b=base_emb, bases_n=attended_novel_emb)
        # sim_maps_softmax = F.softmax(sim_maps, dim=1)
        # max_indices = torch.argmax(sim_maps_softmax, dim=1) # [Bxhxw]
        # new_features = torch.matmul(units.transpose(1, 2), sim_maps_softmax)  # [BxCxN]
        # new_features = new_features.view(B, C, h, w)
        
        # out_fg_n, _, _ = self.orthogonal_decompose(features_n.flatten(2), novel_emb)
        # out_fg_b = out_fg[:, :self.n_base] # [BxKbxCxN]
        # out_fg_n = out_fg[:, self.n_base:] # [BxKnxCxN]
        # outputs = torch.cat([feats_bg, out_fg], dim=1) # [Bx(Kbase+Knovel+1)xCxN]
        '''sim_maps = []
        for i in range(n_class):
            sim_map = F.cosine_similarity(outputs[:, i], features_full, dim=1) # [BxN]
            sim_maps.append(sim_map)
        sim_maps = torch.stack(sim_maps, dim=1) # [Bx(Kbase+Knovel+1)xN]'''
        # sim_maps_reshape = sim_maps.view(B, n_class, h, w) # [Bx(Kbase+Knovel+1)xhxw]

        # out_fg_b = out_fg_b.reshape(B*self.n_base, C, h, w) # [(BxKb)xCxhxw]
        feats_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = feats_b.reshape(B*(self.n_base+1), C, h, w) # [(Bx(1+Kb))xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, self.n_base+1, h, w) # [Bx(Kb+1)xhxw]
        # preds1_softmax = F.softmax(preds1, dim=1)
        # preds1_maxidx = torch.argmax(preds1_softmax, dim=1) # [Bxhxw]
        # n_bg_mask = (preds1_maxidx == 0).float() # [Bxhxw]
        # features_n_bg = features * n_bg_mask.unsqueeze(1) # [BxCxhxw]
        # out_fg_n, pure_bg, _ = self.orthogonal_decompose(features_n_bg.flatten(2), bases_n=novel_emb)



        # feats_n = torch.cat([pure_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        #print(out_fg_n.shape, B, C)
        feats_n = out_fg_n.reshape(B*(self.n_novel), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        preds2 = self.bias_linear(preds2)
        preds = torch.cat([preds1, preds2], dim=1) # [Bx(Kbase+Knovel+1)xhxw
        # bg = 0.5 * preds1[:,0].unsqueeze(1) + 0.5 * preds2[:,0].unsqueeze(1) # [Bx1xhxw]
        # preds = torch.cat([bg, preds1[:,1:], preds2[:,1:]], dim=1)
        # preds = torch.cat([preds2[:,0].unsqueeze(1), preds1[:,1:], preds2[:,1:]], dim=1)
        return preds

    def forward_base(self, img, mask=None, img_b=None, img_s=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # x4, x3, _  = self.backbone.base_forward(img, return_list=True)
        # features = self.backbone.base_forward(img)
        # features = self.decoder(features) # [BxCxhxw]
        img_full = torch.cat([img, img_b, img_s], dim=0)
        features_full = self.backbone.base_forward(img_full)
        # print('features:', features_full.shape)
        features = self.decoder(features_full) # [BxCxhxw]

        B, C, h, w = features.shape
        cls_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]

        n_class = 1 + cls_emb.shape[1]
        features = features.flatten(2) # [BxCxN]
        feats_fg, feats_bg, sim_maps, units = self.orthogonal_decompose(features, cls_emb)
        sim_maps_reshape = sim_maps.view(B, n_class, h, w) # [Bx(Kbase+1)xhxw]
        '''sim_maps_softmax = F.softmax(sim_maps, dim=1)
        # max_indices = torch.argmax(sim_maps_softmax, dim=1) # [Bxhxw]
        new_features = torch.matmul(units.transpose(1, 2), sim_maps_softmax)  # [BxCxN]
        new_features = new_features.view(B, C, h, w)
        preds = self.classifier(new_features) # [Bx1xhxw]'''
        '''s_features = self.backbone.base_forward(s_img)
        s_features = self.decoder(s_features) # [BxCxhxw]
        proto_avg = []
        # print('n_base:', self.n_base, 'base_classes:', self.base_classes)
        for i in range(self.n_base):
            per_base_feats = s_features[i*self.shot:(i+1)*self.shot]
            per_base_labels = s_mask[i*self.shot:(i+1)*self.shot]
            protos = []
            for j in range(self.shot):
                feat = per_base_feats[j].unsqueeze(0)
                # print('feat:', feat.shape) 1,512,60,60
                lbl = per_base_labels[j].unsqueeze(0)
                # print('lbl:', lbl.shape) 1,473,473
                lbl = (lbl == self.base_classes[i]).float()
                lbl = F.interpolate(lbl.unsqueeze(0), size=feat.size()[-2:], mode='nearest').squeeze(0)
                proto = self.Weighted_GAP(feat, lbl)
                protos.append(proto.squeeze(-1).squeeze(-1))
            proto_avg.append(torch.mean(torch.cat(protos, dim=0), dim=0))
        proto_avg = torch.stack(proto_avg, dim=0)  # [KbasexC]
        s_proto = proto_avg.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, h*w) 
        feats_fg_combine = torch.cat([feats_fg, s_proto], dim=2) # [Bx(Kbase)x(C+D)xN]
        feats_fg_combine = feats_fg_combine.view(B*self.n_base, C+s_proto.shape[2], h, w)
        feats_fg_combine = self.combine(feats_fg_combine) # [(BxKbase)xCxhxw]'''

        
        feats_all = torch.cat([feats_bg, feats_fg], dim=1) # [Bx(1+K)xCxN]
        feats_all = feats_all.contiguous().view(B*n_class, C, h, w) # [(B(1+k))xCxhxw]
        preds = self.classifier(feats_all) # [(Bx(1+K))x1xhxw]
        preds = preds.view(B, n_class, h, w) # [Bx(1+K)xhxw]
        preds_u_w = preds[B//3:B//3*2] 
        preds_u_s = preds[B//3*2:]
        # preds_u_norm = F.normalize(preds_u, p=2, dim=1)
        pseudo_label = F.softmax(preds_u_w, dim=1) # [Bxcxhxw]
        max_indices = torch.argmax(pseudo_label, dim=1) # [Bxhxw]
        preds_l = preds[:B//3]
        sim_maps_l = sim_maps[:B//3]
        # cosine_sim = F.cosine_similarity(sim_maps, preds, dim=1) # [Bxhxw]
        # assist_loss = 1 - cosine_sim.mean()

        if self.criterion is not None and mask is not None:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1).squeeze(0) # [KbasexC]
            proto_sim = torch.matmul(cls_emb, cls_emb.t()) # [KbasexKbase]
            assist_loss = self.seg_loss(preds_u_s.to(torch.float), max_indices.to(torch.long))

            return self.criterion(preds_l.to(torch.float), mask, proto_sim=proto_sim), assist_loss
            # return self.criterion(preds, mask, proto_sim=proto_sim, aux_preds=sim_maps_reshape)
        else:
            return preds_l

    def forward_novel(self, img, mask, img_b=None, img_s=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # with torch.no_grad():
        img_full = torch.cat([img, img_b, img_s], dim=0)
        features_full = self.backbone.base_forward(img_full)
        # print('features:', features_full.shape)
        features = self.decoder(features_full) # [BxCxhxw]
        # print('features_b:', features_b.shape)
        # features_n = self.decoder_n(features_full) # [BxCxhxw]

        B, C, h, w = features.shape
        #base_emb = torch.stack([param for param in self.base_emb])
        #novel_emb = torch.stack([param for param in self.novel_emb])
        base_emb = self.base_emb # [KbasexC]
        novel_emb = self.novel_emb # [KnovelxC]
        '''base_emb_exp = base_emb.unsqueeze(0)
        novel_emb_exp = novel_emb.unsqueeze(1)
        cos_sim = F.cosine_similarity(base_emb_exp, novel_emb_exp, dim=-1)  # [5,15]
        attention_weights = F.softmax(cos_sim, dim=-1)
        attended_novel_emb = torch.matmul(attention_weights, base_emb)
        attended_novel_emb = attended_novel_emb.unsqueeze(0) # [1xKnovelxC]'''
        # base to novel attention
        K_b = self.k_linear_b(base_emb)
        Q_n = self.q_linear_n(novel_emb)
        V_b = self.v_linear_b(base_emb)
        # 步骤1: 计算点积得分
        # torch.matmul 可以用来计算矩阵乘法。这里我们需要将 novel_units 与 base_units 的转置相乘。
        scores_b_n = torch.matmul(Q_n, K_b.transpose(0, 1))  # 形状为 [5, 15]

        # 步骤2: 应用 softmax 正规化
        # 对 scores 的最后一个维度（即对每个 novel unit 对所有 base units 的分数）进行 softmax
        attention_weights = F.softmax(scores_b_n, dim=-1)  # 形状仍为 [5, 15]

        # 步骤3: 计算加权和
        # 使用扩展的 attention weights 乘以 base_units
        attended_novel_emb = torch.matmul(attention_weights, V_b)  # 结果形状为 [5, 512]
        attended_novel_emb = attended_novel_emb.unsqueeze(0)
        # # novel to base attention
        # K_n = self.k_linear_n(novel_emb)
        # Q_b = self.q_linear_b(base_emb)  # [KbasexC]
        # V_n = self.v_linear_n(novel_emb) # [KnovelxC]
        # scores_n_b = torch.matmul(Q_b, K_n.transpose(0, 1))  # 形状为 [15, 5]
        # attention_weights_n_b = F.softmax(scores_n_b, dim=-1)  # 形状仍为 [15, 5]
        # attended_base_emb = torch.matmul(attention_weights_n_b, V_n)  # 结果形状为 [15, 512]

        # base_emb = torch.cat([base_emb, attended_base_emb], dim=1) # [Kbasex(C+C)]
        # base_emb = self.fuse_linear_b(base_emb)
        # novel_emb = torch.cat([novel_emb, attended_novel_emb], dim=1) # [Knx(C+C)]
        # novel_emb = self.fuse_linear_n(novel_emb)
        #print(Q.shape,attended_novel_emb.shape)
        base_emb = base_emb.unsqueeze(0) # [1xKbasexC]
        # attended_novel_emb = attended_novel_emb.unsqueeze(0) # [1xKnovelxC]
        # novel_emb = novel_emb.unsqueeze(0) # [1xKnovelxC]
        # novel_emb = torch.cat([novel_emb, attended_novel_emb], dim=1) # [Knx(C+C)]
        # novel_emb = self.fuse_linear(novel_emb)
        novel_emb = novel_emb.unsqueeze(0) # [1xKnovelxC]
        # novel_emb = 0.5 * novel_emb + 0.5 * attended_novel_emb
        # cls_emb = self.cls_emb.unsqueeze(0) # [1x1xC]
        # cls_emb = torch.cat([novel_emb, attended_novel_emb], dim=1) # [1x(Kbase+Knovel)xC]

        n_class = 1 + base_emb.shape[1] + novel_emb.shape[1]
        
        out_fg_b, out_fg_n, feats_bg, sim_maps = self.orthogonal_decompose(features.flatten(2), bases_b=base_emb, bases_n=attended_novel_emb)
        # sim_maps_softmax = F.softmax(sim_maps, dim=1)
        # max_indices = torch.argmax(sim_maps_softmax, dim=1) # [Bxhxw]
        # new_features = torch.matmul(units.transpose(1, 2), sim_maps_softmax)  # [BxCxN]
        # new_features = new_features.view(B, C, h, w)
        
        # out_fg_n, _, _ = self.orthogonal_decompose(features_n.flatten(2), novel_emb)
        # out_fg_b = out_fg[:, :self.n_base] # [BxKbxCxN]
        # out_fg_n = out_fg[:, self.n_base:] # [BxKnxCxN]
        # outputs = torch.cat([feats_bg, out_fg], dim=1) # [Bx(Kbase+Knovel+1)xCxN]
        '''sim_maps = []
        for i in range(n_class):
            sim_map = F.cosine_similarity(outputs[:, i], features_full, dim=1) # [BxN]
            sim_maps.append(sim_map)
        sim_maps = torch.stack(sim_maps, dim=1) # [Bx(Kbase+Knovel+1)xN]'''
        # sim_maps_reshape = sim_maps.view(B, n_class, h, w) # [Bx(Kbase+Knovel+1)xhxw]

        # out_fg_b = out_fg_b.reshape(B*self.n_base, C, h, w) # [(BxKb)xCxhxw]
        feats_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = feats_b.reshape(B*(self.n_base+1), C, h, w) # [(Bx(1+Kb))xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, self.n_base+1, h, w) # [Bx(Kb+1)xhxw]
        # preds1_softmax = F.softmax(preds1, dim=1)
        # preds1_maxidx = torch.argmax(preds1_softmax, dim=1) # [Bxhxw]
        # n_bg_mask = (preds1_maxidx == 0).float() # [Bxhxw]
        # features_n_bg = features * n_bg_mask.unsqueeze(1) # [BxCxhxw]
        # out_fg_n, pure_bg, _ = self.orthogonal_decompose(features_n_bg.flatten(2), bases_n=novel_emb)



        # feats_n = torch.cat([pure_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        #print(out_fg_n.shape, B, C)
        feats_n = out_fg_n.reshape(B*(self.n_novel), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        preds2 = self.bias_linear(preds2)
        preds = torch.cat([preds1, preds2], dim=1) # [Bx(Kbase+Knovel+1)xhxw
        # bg = 0.5 * preds1[:,0].unsqueeze(1) + 0.5 * preds2[:,0].unsqueeze(1) # [Bx1xhxw]
        # preds = torch.cat([bg, preds1[:,1:], preds2[:,1:]], dim=1)
        # preds = torch.cat([preds2[:,0].unsqueeze(1), preds1[:,1:], preds2[:,1:]], dim=1)
        # preds = torch.cat([preds1, preds2], dim=1)  # [Bx(Kbase+Knovel+1)xhxw]
        # preds = self.classifier(feats_n)
        # preds = preds.view(B, 1+self.n_novel+self.n_base, h, w)
        # preds2 = torch.cat([preds[:,0].unsqueeze(1), preds[:,1+self.n_base:]], dim=1) # [Bx(1+Kn)xhxw]
        # logits = preds[:, 1:] # [Bx(Kbase+Knovel)xhxw]
        # sim_maps = sim_maps.view(B, n_class, h, w)
        # sim_maps_u_norm = F.normalize(sim_maps_u, p=2, dim=1)
        # soft_sim_maps_u_w = F.softmax(sim_maps_u_w, dim=1)
        # max_indices = torch.argmax(soft_sim_maps_u_w, dim=1) # [Bxhxw] 
        preds_u_w = preds[B//3:B//3*2] 
        preds_u_s = preds[B//3*2:]
        # preds_u_norm = F.normalize(preds_u, p=2, dim=1)
        pseudo_label = F.softmax(preds_u_w, dim=1) # [Bxcxhxw]
        max_indices = torch.argmax(pseudo_label, dim=1) # [Bxhxw]
        # pseudo_mask = (pseudo_label.max(1)[0] >= 0.5).float() # [Bxhxw]
        # # 转换为one-hot编码
        # preds_one_hot = F.one_hot(max_indices, num_classes=C)  # 生成的形状为[B, H, W, C]
        # # 由于one_hot最后一个维度是C，我们需要将其移至第二个位置
        # preds_one_hot = preds_one_hot.permute(0, 3, 1, 2)  # 调整维度，最终形状为[B, C, H, W]
        # # one_hot现在是整数类型，你可能需要将其转换为与soft_preds_u相同的数据类型
        # preds_one_hot = preds_one_hot.float()
        # assist_loss = self.mse_loss(soft_sim_maps_u, soft_preds_u)
        # soft_sim_maps = F.softmax(sim_maps, dim=1)
        # soft_preds = F.softmax(preds, dim=1)
        # max_indices = torch.argmax(soft_preds, dim=1) # [Bxhxw]
        preds_l = preds[:B//3]
        sim_maps_l = sim_maps[:B//3]
        # cosine_sim = F.cosine_similarity(sim_maps, preds, dim=1) # [Bxhxw]
        # assist_loss = 1 - cosine_sim.mean()

        if self.criterion is not None and mask is not None:
            with torch.cuda.amp.autocast(enabled=False):
                # mask_all = torch.cat([mask, mask_new], dim=0)
                # novel_emb = torch.cat([novel_emb, bg_emb], dim=1) # [1x(Kn+1)xC]
                # novel_emb = cls_emb[self.n_base:] # [1xKnovelxC]
                novel_emb = F.normalize(novel_emb.to(torch.float), p=2, dim=-1) # [1xKnovelxC]
                novel_emb = novel_emb.reshape(-1, C) # [KnxC]
                # base_emb = cls_emb[:self.n_base]
                all_emb = torch.cat([novel_emb, F.normalize(base_emb.to(torch.float).squeeze(0), p=2, dim=-1)], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]
                # scale_maps = F.interpolate(input=sim_maps, size=mask.shape[1:], mode='bilinear', align_corners=True)
                assist_loss = self.seg_loss(preds_u_s.to(torch.float), max_indices.to(torch.long))
                # assist_loss = torch.mean(sum_loss * pseudo_mask)

                return self.criterion(preds_l.to(torch.float), mask, is_ft=True, proto_sim=proto_sim), assist_loss
        else:
            return preds_l
