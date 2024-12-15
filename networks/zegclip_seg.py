import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .clip import VPTCLIPVisionTransformer, CLIPTextEncoder

from .layers import FPN, Projector, ATMSingleHeadSeg


class ZegCLIP(nn.Module):
    def __init__(self, base_class, img_size, patch_size, width, get_embeddings, drop_path_rate, layers, num_tokens, prompt_dim, total_d_layer, clip_pretrain=None, load_text_embedding=None, word_len=77, vis_dim=512, out_indices=[11]):
        super().__init__()
        # Vision & Text Encoder
        self.img_encoder = VPTCLIPVisionTransformer(input_resolution=img_size, 
                                                 patch_size=patch_size, 
                                                 width=width, 
                                                 layers=layers,
                                                 get_embeddings=get_embeddings, 
                                                 drop_path_rate=drop_path_rate,
                                                 out_indices=out_indices, 
                                                 num_tokens=num_tokens, 
                                                 prompt_dim=prompt_dim, 
                                                 total_d_layer=total_d_layer, 
                                                 pretrained=clip_pretrain)
        self.text_encoder = CLIPTextEncoder(context_length=word_len, 
                                            embed_dim=vis_dim,
                                            pretrained=clip_pretrain)
        self.load_text_embedding = load_text_embedding
        self.base_class = base_class
        

    def forward(self, img, word):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        
        vis = self.img_encoder(img)
        if self.load_text_embedding:
            text_feat = np.load(self.load_text_embedding)
            text_feat = torch.from_numpy(text_feat).to(img.device)
            # print('text_feat:', text_feat.size()) 20, 512
        text_feat = text_feat[self.base_class, :]  # [15, 512]
        # print(self.base_class) 15
        feat = []
        feat.append(vis)
        feat.append(text_feat)

        return feat

class GFSS_Model(nn.Module):
    '''
        Segmenter for Generalized Few-shot Semantic Segmentation
    '''
    def __init__(self, fold, n_base, criterion=None, norm_layer=nn.BatchNorm2d, use_base=True, is_ft=False, n_novel=5, img_size=512, patch_size=16, in_channels=512, out_indices=[11], 
                 width=768, get_embeddings=True, drop_path_rate=0.1, layers=12, num_tokens=10, prompt_dim=768, total_d_layer=11, clip_pretrain=None, load_text_embedding=None, word_len=77, 
                 vis_dim=512, num_layers=3, num_head=8, **kwargs):
        super(GFSS_Model, self).__init__()
        num_classes = 20
        d_model = 512
        self.fold = fold
        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'potted-plant',
                       'sheep', 'sofa', 'train', 'tv/monitor']
        if fold == -1:
            # training with all classes
            base_classes = set(range(0, num_classes))
            novel_classes = set()
            
        else:
            interval = num_classes // 4
            # base classes = all classes - novel classes
            base_classes = set(range(0, num_classes)) - set(range(interval * fold, interval * (fold + 1)))
            # novel classes
            novel_classes = set(range(interval * fold, interval * (fold + 1)))
        base_classes = list(base_classes)
        novel_classes = list(novel_classes)

        self.backbone = ZegCLIP(base_classes, img_size[0], patch_size, width, get_embeddings, drop_path_rate, 
                                layers, num_tokens, prompt_dim, total_d_layer, 
                                clip_pretrain, load_text_embedding, word_len, vis_dim, out_indices)
        # Decoder
        self.decoder = ATMSingleHeadSeg(img_size=img_size[0],
                                        in_channels=in_channels,
                                        base_idx=base_classes,
                                        novel_idx=novel_classes,
                                        channels = in_channels,
                                        num_classes=n_base,
                                        num_layers=num_layers,
                                        num_heads=num_head,
                                        use_proj=False,
                                        use_stages=len(out_indices),
                                        embed_dims=in_channels,
                                        criterion=criterion)
        
        self.n_novel = n_novel
        self.use_base = use_base
        self.is_ft = is_ft
        self.criterion = criterion
        self.n_base = n_base


        if self.training:
            self._freeze_stages(self.backbone, exclude_key='prompt')

        else:
            self.backbone.eval()

    def init_cls_n(self):
        for param_q, param_k in zip(self.classifier.parameters(), self.classifier_n.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    def train_mode(self):
        self.train()
        # to prevent BN from learning data statistics with exponential averaging
        self.backbone.eval()
        self.decoder.eval()
        # self.classifier.eval()
    
    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count>0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def ft_freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # for param in self.backbone.neck.parameters():
            # param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # for param in self.classifier.parameters():
            # param.requires_grad = False
    
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

    def forward(self, img, mask=None, text=None, img_b=None, mask_b=None, text_b=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        if self.is_ft:
            # print('training:', self.training)
            if self.training:
                return self.forward_novel(img, mask, text, img_b, mask_b, text_b)
            else:
                return self.forward_all(img, mask, text)
        else:
            return self.forward_base(img, mask, text)

    def forward_all(self, img, mask=None, text=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        features = self.backbone(img, text) # [BxCxhxw]
        # _, _, v5 = features
        # features = torch.cat([features, features_novel], dim=1)
        # features = self.reduce_dim(features)
        features = self.decoder(features) # [BxCxhxw]
        B, C, h, w = features.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        # novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        out_fg_b, feats_bg = self.orthogonal_decompose(features.flatten(2), base_emb)

        out_fg_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = out_fg_b.contiguous().view(B*(1+self.n_base+self.n_novel), C, h, w) # [(BxKb)xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, 1+self.n_base+self.n_novel, h, w) # [BxKbxhxw]
        # preds1 = self.bias_linear(preds1)

        '''feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = feats_n.contiguous().view(B*(1+self.n_novel), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, 1+self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        '''
        # preds2_n = self.bias_linear(preds2)

        # preds = torch.cat([preds1, preds2_n], dim=1)
        # preds = torch.cat([preds2[:,0].unsqueeze(1), preds1, preds2[:,1:]], dim=1)
        # preds = self.bias_linear(preds)
        return preds1

    def forward_base(self, img, mask=None, text=None):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # x4, x3, _  = self.backbone.base_forward(img, return_list=True)
        features = self.backbone(img, text) # [BxCxhxw]
        # _, _, v5 = features
        preds = self.decoder(features, mask) # [Bx15xHxW]
        # print('mask:', mask.size(), 'preds:', preds.size())
        if self.criterion is not None and mask is not None:
            
            loss_dict = self.criterion(preds, mask)
            # print(mask.size(), pred.size())
            '''mask = mask.float().unsqueeze(1)
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)'''
            return loss_dict
        else:
            return preds


    def forward_novel(self, img, mask, text, img_b, mask_b, text_b):
        '''
            img       : [BxCxHxW]
            mask      : [BxHxW]
        '''
        # with torch.no_grad():
        img_full = torch.cat([img, img_b], dim=0)
        text_full = torch.cat([text, text_b], dim=0)
        features_full = self.backbone(img_full, text_full)
        # features_full = torch.cat([features_full, features_novel], dim=1)
        # features_full = self.reduce_dim(features_full)
        features_full = self.decoder(features_full) # [BxCxhxw]

        B, C, h, w = features_full.shape

        base_emb = self.base_emb.unsqueeze(0) # [1xKbasexC]
        # novel_emb = self.novel_emb.unsqueeze(0) # [1xKnovelxC]

        n_class = 1 + base_emb.shape[1] # + novel_emb.shape[1]
        features_full = features_full.flatten(2) # [BxCxN]
        out_fg_b, feats_bg = self.orthogonal_decompose(features_full, base_emb)

        out_fg_b = torch.cat([feats_bg, out_fg_b], dim=1) # [Bx(1+Kb)xCxN]
        feats_b = out_fg_b.reshape(B*(n_class), C, h, w) # [(Bx(1+Kb))xCxhxw]
        preds1 = self.classifier(feats_b) # [(BxKb)x1xhxw]
        preds1 = preds1.view(B, n_class, h, w) # [BxKbxhxw]
        # preds1 = self.bias_linear(preds1)

        '''feats_n = torch.cat([feats_bg, out_fg_n], dim=1) # [Bx(1+Kn)xCxN]
        feats_n = feats_n.reshape(B*(1+self.n_novel), C, h, w) # [(Bx(1+Kn))xCxhxw]
        preds2 = self.classifier_n(feats_n) # [(Bx(1+Kn))x1xhxw]
        preds2 = preds2.view(B, 1+self.n_novel, h, w) # [Bx(1+Kn)xhxw]
        # preds2_n = self.bias_linear(preds2)
        '''
        # preds = torch.cat([preds1, preds2_n], dim=1)
        # preds = torch.cat([preds2[:,0].unsqueeze(1), preds1, preds2[:,1:]], dim=1)

        # relabeling
        mask_new = []
        for b in range(B//2):
            bg_mask = mask_b[b] == 0
            bg_out = preds1[B//2+b] # [(1+Kn)xhxw]
            bg_out = F.interpolate(input=bg_out.unsqueeze(0), size=mask_b[b].shape, mode='bilinear', align_corners=True)

            bg_idx = torch.argmax(bg_out.squeeze(0), dim=0) # [hxw]
            # bg_idx[bg_idx>0] += self.n_base
            mask_b[b][bg_mask] = bg_idx[bg_mask]
            mask_new.append(mask_b[b])
        mask_new = torch.stack(mask_new, dim=0)

        if self.criterion is not None and mask is not None:
            with torch.cuda.amp.autocast(enabled=False):
                mask_all = torch.cat([mask, mask_new], dim=0)
                '''novel_emb = F.normalize(novel_emb.to(torch.float), p=2, dim=-1) # [1xKnovelxC]
                novel_emb = novel_emb.reshape(-1, C) # [KnxC]
                all_emb = torch.cat([novel_emb, F.normalize(base_emb.to(torch.float).squeeze(0), p=2, dim=-1)], dim=0) # [((Kn+Kb)xC]
                proto_sim = torch.matmul(novel_emb, all_emb.t()) # [Knx(Kn+Kb)]'''
                base_emb = F.normalize(base_emb, p=2, dim=-1).squeeze(0) # [KbasexC]
                proto_sim = torch.matmul(base_emb, base_emb.t()) # [KbasexKbase]
                loss_dict = self.criterion(preds1.to(torch.float), mask_all, is_ft=True, proto_sim=proto_sim)
                return loss_dict
            # mask = mask.float().unsqueeze(1)
            '''if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)'''
        else:
            return preds