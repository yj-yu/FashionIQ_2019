import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
from model.net_vlad import NetVLAD
from model.composition import TIRG, AttrMLB, NormalizationLayer, TrgTIRG
from model.effnet import EffNet
from model.film import FilmResBlock
from model.fusion import CompactBilinearPooling
from base import BaseModel
import pretrainedmodels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CENet(BaseModel):
    def __init__(self, text_dim, use_ce, use_film, vlad_clusters, composition, target_comp, fusion, attr_fusion,
                 expert_dims, same_dim, text_feat, norm_scale=5.0, vocab_size=None, we_parameter=None,
                 attr_vocab_size=None, mimic_ce_dims=False, concat_experts=False, concat_mix_experts=False,
                 backbone='resnet'):
        super().__init__()

        self.composition_type = composition
        self.fusion = fusion
        self.expert_dims = expert_dims
        self.text_feat = text_feat
        self.vocab_size = vocab_size

        if text_feat == 'learnable':
            self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
            if we_parameter is not None:
                self.embed.weight.data.copy_(torch.from_numpy(we_parameter))

            text_pooling_list = [
                [mode, TextMultilevelEncoding(word_dim=text_dim, hidden_dim=text_dim)] for mode in ('src', 'trg')]
            self.text_pooling = nn.ModuleDict(text_pooling_list)

            # self.src_text_pooling = TextMultilevelEncoding(
            #     word_dim=text_dim,
            #     hidden_dim=text_dim
            # )
            # self.trg_text_pooling = TextMultilevelEncoding(
            #     word_dim=text_dim,
            #     hidden_dim=text_dim
            # )
            encoder_text_dim = text_dim + vocab_size + 1536
        elif text_feat == 'w2v':
            text_pooling_list = [
                [mode, NetVLAD(feature_size=text_dim, cluster_size=vlad_clusters["text"])] for mode in ('src', 'trg')]
            self.text_pooling = nn.ModuleDict(text_pooling_list)

            # self.src_text_pooling = NetVLAD(
            #     feature_size=text_dim,
            #     cluster_size=vlad_clusters["text"],
            # )
            # self.trg_text_pooling = NetVLAD(
            #     feature_size=text_dim,
            #     cluster_size=vlad_clusters["text"],
            # )
            # encoder_text_dim = self.src_text_pooling.out_dim
            encoder_text_dim = self.text_pooling['src'].out_dim
        else:
            raise ValueError

        text_encoder_list = [
            [mode,
             TextCEModule(expert_dims=expert_dims,
                          text_dim=encoder_text_dim,
                          concat_experts=concat_experts,
                          concat_mix_experts=concat_mix_experts,
                          same_dim=same_dim)] for mode in ('src', 'trg')]
        # text_encoder_list = [
        #     ['src',
        #      TextCEModule(expert_dims=expert_dims,
        #                   text_dim=encoder_text_dim,
        #                   concat_experts=concat_experts,
        #                   concat_mix_experts=concat_mix_experts,
        #                   same_dim=same_dim)],
        #     ['trg',
        #      TextCEModule(expert_dims=expert_dims,
        #                   text_dim=300,
        #                   concat_experts=concat_experts,
        #                   concat_mix_experts=concat_mix_experts,
        #                   same_dim=same_dim)]]

        self.text_encoder = nn.ModuleDict(text_encoder_list)

        # self.src_text_encoder = TextCEModule(
        #     expert_dims=expert_dims,
        #     text_dim=encoder_text_dim,
        #     concat_experts=concat_experts,
        #     concat_mix_experts=concat_mix_experts,
        #     same_dim=same_dim
        # )
        # self.trg_text_encoder = TextCEModule(
        #     expert_dims=expert_dims,
        #     text_dim=encoder_text_dim,
        #     concat_experts=concat_experts,
        #     concat_mix_experts=concat_mix_experts,
        #     same_dim=same_dim
        # )

        self.image_encoder = VideoCEModule(
            backbone=backbone,
            use_ce=use_ce,
            concat_experts=concat_experts,
            concat_mix_experts=concat_mix_experts,
            expert_dims=expert_dims,
            mimic_ce_dims=mimic_ce_dims,
            same_dim=same_dim,
            vlad_clusters=vlad_clusters,
            attr_vocab_size=attr_vocab_size,
            attr_fusion_name=attr_fusion
        )

        if self.composition_type == 'multi':
            composition_list = [TIRG(fusion, embed_dim=same_dim) for _ in self.expert_dims]
            self.composition_layer = nn.ModuleList(composition_list)
        else:
            self.composition_layer = TIRG(fusion, embed_dim=same_dim)

        self.normalization_layer = NormalizationLayer(normalize_scale=norm_scale, learn_scale=True)
        self.trg_normalization_layer = NormalizationLayer(normalize_scale=norm_scale, learn_scale=True)

        if target_comp == 'cnn':
            target_comp_list = [CNNAttention() for _ in self.expert_dims]
            self.target_composition = nn.ModuleList(target_comp_list)
        elif target_comp == 'film':
            target_comp_list = [DualTIRG2Film() for _ in self.expert_dims]
            self.target_composition = nn.ModuleList(target_comp_list)
        elif target_comp == 'tirg':
            target_comp_list = [TIRG(fusion, embed_dim=same_dim) for _ in self.expert_dims]
            self.target_composition = nn.ModuleList(target_comp_list)
        elif target_comp == 'ba':
            target_comp_list = [BilinearAttention(dim=same_dim) for _ in self.expert_dims]
            self.target_composition = nn.ModuleList(target_comp_list)
        elif target_comp == 'cbpa':
            target_comp_list = [CBPAttention(dim=same_dim) for _ in self.expert_dims]
            self.target_composition = nn.ModuleList(target_comp_list)
        else:
            raise ValueError

    def get_text_feature(self, text, ind, text_bow=None, text_lengths=None, target=False):
        mode = 'trg' if target else 'src'

        if self.text_feat == 'learnable':
            text = self.embed(text)

        batch_size, max_words, text_feat_dim = text.size()
        captions_per_video = 1

        if self.text_feat == 'learnable':
            text = self.text_pooling[mode](text, text_bow, text_lengths)
        elif self.text_feat == 'w2v':
            text = self.text_pooling[mode](text)

        text = text.view(batch_size, captions_per_video, -1)
        text, moe_weights = self.text_encoder[mode](text, ind, target)

        return text, moe_weights

    def get_combined_feature(self, experts, text, target=False):
        mode = 'trg' if target else 'src'

        composition_feature = {}
        if mode == 'src':
            for modality, layer in zip(self.expert_dims.keys(), self.composition_layer):
                composition_feature[modality] = \
                    self.normalization_layer(layer(experts[modality], text[modality])).unsqueeze(1)
        else:
            for modality, layer in zip(self.expert_dims.keys(), self.target_composition):
                composition_feature[modality] = \
                    self.trg_normalization_layer(layer(experts[modality], text[modality])).unsqueeze(1)

        return composition_feature

    def forward(self, experts, ind, text=None, text_bow=None, text_lengths=None, raw_captions=None, target=False):
        experts = self.image_encoder(experts, ind)

        if text is not None:
            if target:
                if self.text_feat == 'learnable':
                    text = self.embed(text)
                batch_size, max_words, text_feat_dim = text.size()
                # TODO: multiple caption
                captions_per_video = 1
                
                if self.text_feat == 'learnable':
                    text = self.trg_text_pooling(text, text_bow, text_lengths)
                elif self.text_feat == 'w2v':
                    text = self.trg_text_pooling(text)
                
                text = text.view(batch_size, captions_per_video, -1)
                text, moe_weights = self.trg_text_encoder(text, ind)

                composition_feature = {}
                for i, modality in enumerate(self.expert_dims.keys()):
                    composition_feature[modality] = \
                        self.normalization_layer(self.target_composition[i](experts[modality], text[modality])).unsqueeze(1)
            else:
                if self.text_feat == 'learnable':
                    text = self.embed(text)
                batch_size, max_words, text_feat_dim = text.size()
                captions_per_video = 1

                if self.text_feat == 'learnable':
                    text = self.src_text_pooling(text, text_bow, text_lengths)
                elif self.text_feat == 'w2v':
                    text = self.src_text_pooling(text)

                text = text.view(batch_size, captions_per_video, -1)
                text, moe_weights = self.src_text_encoder(text, ind)

                composition_feature = {}

                if self.composition_type == 'multi':
                    for modality, layer in zip(self.expert_dims.keys(), self.composition_layer):
                        composition_feature[modality] = \
                            self.normalization_layer(layer(experts[modality], text[modality])).unsqueeze(1)
                elif self.composition_type == 'single':
                    for modality in self.expert_dims.keys():
                        composition_feature[modality] = \
                            self.normalization_layer(self.composition_layer(experts[modality], text[modality])).unsqueeze(1)
                else:
                    raise ValueError
        elif text is None and target:
            composition_feature = experts
            moe_weights = None
        else:
            for modality in self.expert_dims.keys():
                experts[modality] = self.normalization_layer(experts[modality])
            composition_feature = experts
            moe_weights = None

        return composition_feature, text, moe_weights


class DualTIRG2Film(nn.Module):
    def __init__(self, dim=512):
        super(DualTIRG2Film, self).__init__()
        self.dim = dim
        self.film1 = FilmResBlock(dim, dim)
        self.film2 = FilmResBlock(dim, dim)

    def forward(self, img, text):
        img = img.view(-1, self.dim, 1, 1)
        img = self.film1(img, text.view(-1, self.dim))
        img = self.film2(img, text.view(-1, self.dim))
        img = img.view(-1, self.dim)

        return img


class CNNAttention(nn.Module):
    def __init__(self, dim=512):
        super(CNNAttention, self).__init__()
        self.dim = 512
        self.cnn = nn.Conv2d(dim, dim, 1)

        # self.text_net = nn.Sequential(
        #     nn.Linear(self.dim, self.dim),
        #     nn.Sigmoid()
        # )

    def forward(self, img, text):
        img = img.view(-1, self.dim, 1, 1)
        img = self.cnn(img)
        img = img.view(-1, self.dim)
        # attention = text.view(-1, self.dim).softmax(-1)
        # attention = torch.sigmoid(text.view(-1, self.dim))
        text = text / 500
        attention = text.view(-1, self.dim).softmax(-1)
        # attention = torch.tanh(text.view(-1, self.dim))
        # attention = self.text_net(text.view(-1, self.dim))
        img_att = img * attention
        img = img + img_att

        return img


class BilinearAttention(nn.Module):
    def __init__(self, dim=512):
        super(BilinearAttention, self).__init__()
        self.dim = dim
        self.img_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh())

        self.text_net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(dim, dim),
            nn.Tanh())

        self.output_net = nn.Linear(dim, dim)

    def forward(self, img, text):
        img = img.view(-1, self.dim)
        text = text.view(-1, self.dim)
        hadamard = self.img_net(img) * self.text_net(text)
        # attention = self.output_net(hadamard).sigmoid()
        attention = self.output_net(hadamard).softmax(-1)
        att_img = img * attention
        # return att_img
        img = img + att_img

        return img


class CBPAttention(nn.Module):
    def __init__(self, dim=512):
        super(CBPAttention, self).__init__()
        self.dim = dim
        self.cbp = CompactBilinearPooling(dim, dim, sum_pool=False)

    def forward(self, img, text):
        img = img.view(-1, self.dim)
        text = text.view(-1, self.dim)
        cbp = self.cbp(img, text)
        cbp = cbp / 500
        attention = cbp.softmax(-1)
        att_img = img * attention
        img = img + att_img

        return img


class TextCEModule(nn.Module):
    def __init__(self, expert_dims, text_dim, concat_experts, concat_mix_experts, same_dim=512):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.concat_experts = concat_experts
        self.concat_mix_experts = concat_mix_experts
        self.moe_fc = nn.Linear(text_dim, len(expert_dims))
        num_mods = len(expert_dims)
        self.moe_weights = torch.ones(1, num_mods) / num_mods

        agg_dims = [expert_dims[mod][1] for mod in modalities]
        text_out_dims = [same_dim for _ in agg_dims]

        if self.concat_experts:
            gated_text_embds = [nn.Sequential()]
        elif self.concat_mix_experts:
            gated_text_embds = [GatedEmbeddingUnit(text_dim, sum(agg_dims), use_bn=True)]
        else:
            gated_text_embds = [GatedEmbeddingUnit(text_dim, dim, use_bn=True) for dim in text_out_dims]

        self.text_GU = nn.ModuleList(gated_text_embds)

    def compute_moe_weights(self, text, ind):
        B, K, D = text.shape
        M = len(self.modalities)

        text = text.view(B * K, D)
        moe_weights = self.moe_fc(text)  # BK x D -> BK x M
        moe_weights = F.softmax(moe_weights, dim=1)
        moe_weights = moe_weights.view(B, K, M)
        available = torch.zeros_like(moe_weights)

        for ii, modality in enumerate(self.modalities):
            available[:, :, ii] = ind[modality].view(-1, 1).repeat(1, K)

        msg = "expected `available` modality mask to only contain 0s or 1s"
        assert set(torch.unique(available).cpu().numpy()).issubset(set([0, 1])), msg

        moe_weights = available * moe_weights
        norm_weights = torch.sum(moe_weights, dim=2)
        norm_weights = norm_weights.view(B, K, 1)
        moe_weights = torch.div(moe_weights, norm_weights)  # B x K x M

        return moe_weights

    def forward(self, text, ind, target):
        text_embd = {}

        B, captions_per_video, feat_dim = text.size()
        text = text.view(B * captions_per_video, feat_dim)

        for modality, layer in zip(self.modalities, self.text_GU):
            text_ = layer(text)
            text_ = text_.view(B, captions_per_video, -1)
            text_embd[modality] = text_
        text = text.view(B, captions_per_video, -1)

        if target:
            moe_weights = None
        else:
            moe_weights = self.compute_moe_weights(text, ind)

        return text_embd, moe_weights


class VideoCEModule(nn.Module):
    def __init__(self, backbone, expert_dims, use_ce, mimic_ce_dims, concat_experts, concat_mix_experts,
                 vlad_clusters, attr_fusion_name, attr_vocab_size, same_dim=512):
        super().__init__()

        modalities = list(expert_dims.keys())
        self.expert_dims = expert_dims
        self.modalities = modalities
        self.use_ce = use_ce
        self.mimic_ce_dims = mimic_ce_dims
        self.concat_experts = concat_experts
        self.concat_mix_experts = concat_mix_experts
        self.attr_fusion_name = attr_fusion_name
        self.backbone_name = backbone

        in_dims = [expert_dims[mod][0] for mod in modalities]
        agg_dims = [expert_dims[mod][1] for mod in modalities]
        use_bns = [True for modality in self.modalities]

        if self.use_ce or self.mimic_ce_dims:
            dim_reducers = [ReduceDim(in_dim, same_dim) for in_dim in in_dims]
            self.video_dim_reduce = nn.ModuleList(dim_reducers)

        if self.use_ce:
            self.g_reason_1 = nn.Linear(same_dim * 2, same_dim)
            self.g_reason_2 = nn.Linear(same_dim, same_dim)

            self.f_reason_1 = nn.Linear(same_dim, same_dim)
            self.f_reason_2 = nn.Linear(same_dim, same_dim)

            gated_vid_embds = [GatedEmbeddingUnitReasoning(same_dim) for _ in in_dims]

        elif self.mimic_ce_dims:  # ablation study
            gated_vid_embds = [MimicCEGatedEmbeddingUnit(same_dim, same_dim, use_bn=True) for _ in modalities]

        elif self.concat_mix_experts:  # ablation study
            in_dim, out_dim = sum(in_dims), sum(agg_dims)
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, out_dim, use_bn=True)]

        elif self.concat_experts:  # ablation study
            gated_vid_embds = []

        else:
            gated_vid_embds = [GatedEmbeddingUnit(in_dim, dim, use_bn) for
                               in_dim, dim, use_bn in zip(in_dims, agg_dims, use_bns)]

        self.video_GU = nn.ModuleList(gated_vid_embds)

        if backbone == 'resnet':
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
        elif backbone == 'densenet':
            densenet = models.densenet169(pretrained=True)
            modules = list(densenet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
        elif backbone in ['inceptionresnetv2', 'pnasnet5large', 'nasnetalarge', 'senet154', 'polynet']:
            self.backbone = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        else:
            raise ValueError
        self.dropout = nn.Dropout(p=0.2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # self.video_multi_encoding = VideoMultilevelEncoding(in_dim=in_dims[-1], out_dim=in_dims[-1])

        if 'keypoint' in self.expert_dims.keys():
            self.effnet = EffNet()
            self.keypoint_pooling = NetVLAD(
                feature_size=512,
                cluster_size=vlad_clusters['keypoint'],
            )

        if 'attr0' in self.expert_dims.keys():
            self.attr_embed = nn.Embedding(attr_vocab_size, 300, padding_idx=0)
            attr_pooling_list = [NetVLAD(feature_size=300, cluster_size=vlad_clusters['attr']) for _ in range(6)]
            self.attr_pooling = nn.ModuleList(attr_pooling_list)

            if attr_fusion_name == 'attrmlb':
                self.attr_fusion = AttrMLB()
            else:
                self.attr_fusion = TIRG(attr_fusion_name, embed_dim=same_dim)

    def forward(self, experts, ind):
        B = experts[self.modalities[0]].size(0)
        if self.backbone_name in ["resnet", "densenet"]:
            image_feature = self.backbone(experts['resnet'])
        else:
            image_feature = self.backbone.features(experts['resnet'])

        for key, val in experts.items():
            if key == 'resnet':
                experts[key] = self.dropout(self.avg_pooling(image_feature).squeeze(-1).squeeze(-1))
                # experts[key] = self.video_multi_encoding(image_feature)
            if key == 'keypoint':
                val = val.view(-1, 3, 40, 40)
                keypoint_feature = self.effnet(val.float()).view(B, -1, 512)
                experts[key] = self.keypoint_pooling(keypoint_feature)

        if 'attr0' in self.expert_dims.keys():
            for i in range(6):
                val = drop_nans(x=experts[f'attr{i}'], ind=ind[f'attr{i}'])
                val = self.attr_embed(val.long())
                val = self.attr_pooling[i](val)
                experts[f'attr{i}'] = val

        if 'attr0' in self.expert_dims.keys() and self.attr_fusion_name == 'attrmlb':
            for i in range(6):
                experts[f'attr{i}'] = self.attr_fusion(image_feature, experts[f'attr{i}'])

        if hasattr(self, "video_dim_reduce"):
            # Embed all features to a common dimension
            for modality, layer in zip(self.modalities, self.video_dim_reduce):
                experts[modality] = layer(experts[modality])

        if 'attr0' in self.expert_dims.keys() and self.attr_fusion_name != 'attrmlb':
            for i in range(6):
                experts[f'attr{i}'] = self.attr_fusion(experts['resnet'], experts[f'attr{i}'])

        if self.use_ce:
            all_combinations = list(itertools.permutations(experts, 2))
            assert len(self.modalities) > 1, "use_ce requires multiple modalities"

            tmp = []

            for ii, l in enumerate(self.video_GU):

                mask_num = 0
                curr_mask = 0
                temp_dict = {}
                avai_dict = {}
                curr_modality = self.modalities[ii]

                # if curr_modality == 'resnet':
                #     continue
                # else:
                for modality_pair in all_combinations:
                    mod0, mod1 = modality_pair
                    if mod0 == curr_modality:
                        new_key = "_".join(modality_pair)
                        fused = torch.cat((experts[mod0], experts[mod1]), 1)  # -> B x 2D
                        temp = self.g_reason_1(fused)  # B x 2D -> B x D
                        temp = self.g_reason_2(F.relu(temp))  # B x D -> B x D
                        temp_dict[new_key] = temp
                        avail = (ind[mod0].float() * ind[mod1].float()).to(device)
                        avai_dict[new_key] = avail

                # Combine the paired features into a mask through elementwise summation
                for mm in temp_dict:
                    curr_mask += temp_dict[mm] * avai_dict[mm].unsqueeze(1)
                    mask_num += avai_dict[mm]

                curr_mask = torch.div(curr_mask, (mask_num + 0.00000000001).unsqueeze(1))
                curr_mask = self.f_reason_1(curr_mask)
                curr_mask = self.f_reason_2(F.relu(curr_mask))
                experts[curr_modality] = l(experts[curr_modality], curr_mask)

            return experts

        elif self.concat_mix_experts:
            concatenated = torch.cat(tuple(experts.values()), dim=1)
            vid_embd_ = self.video_GU[0](concatenated)
            return vid_embd_
        elif self.concat_experts:
            vid_embd_ = torch.cat(tuple(experts.values()), dim=1)
            return vid_embd_
        else:
            for modality, layer in zip(self.modalities, self.video_GU):
                experts[modality] = layer(experts[modality])
            return experts


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super(GatedEmbeddingUnit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        x = F.normalize(x)
        return x


class MimicCEGatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension, use_bn):
        super().__init__()
        self.cg = ContextGating(input_dimension, add_batch_norm=use_bn)

    def forward(self, x):
        x = self.cg(x)
        x = F.normalize(x)
        return x


class ReduceDim(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(ReduceDim, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x1 = self.fc(x)
        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class GatedEmbeddingUnitReasoning(nn.Module):
    def __init__(self, output_dimension):
        super(GatedEmbeddingUnitReasoning, self).__init__()
        self.cg = ContextGatingReasoning(output_dimension)

    def forward(self, x, mask):
        x = self.cg(x, mask)
        x = F.normalize(x)
        return x


class ContextGatingReasoning(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(ContextGatingReasoning, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        self.batch_norm2 = nn.BatchNorm1d(dimension)

    def forward(self, x, x1):
        x2 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1)
            x2 = self.batch_norm2(x2)

        t = x1 + x2
        x = torch.cat((x, t), 1)
        return F.glu(x, 1)


def drop_nans(x, ind):
    missing = torch.nonzero(ind == 0).flatten()
    if missing.numel():
        x_ = x
        x_[missing] = 0
        x = x_
    return x


def xavier_init_fc(fc):
    r = np.sqrt(6.) / np.sqrt(fc.in_features + fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class MFC(nn.Module):
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False, activation=None):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()
        self.activation = activation

    def init_weights(self):
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):
        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)
        if self.activation is not None:
            self.activation(features)
        return features


class VideoMultilevelEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VideoMultilevelEncoding, self).__init__()
        self.visual_mapping = MFC([in_dim, in_dim], 0.2, have_bn=True, have_last_bn=True, activation=F.leaky_relu)
        self.visual_mapping2 = MFC([in_dim, in_dim], 0.2, have_bn=True, have_last_bn=True, activation=F.leaky_relu)
        self.visual_mapping3 = MFC([in_dim, out_dim], 0.2, have_bn=True, have_last_bn=True)

    def forward(self, res_out):
        features = self.visual_mapping(res_out)
        features = self.visual_mapping2(features)
        features = self.visual_mapping3(features)

        return features


class TextMultilevelEncoding(nn.Module):
    def __init__(self, word_dim, hidden_dim):
        super(TextMultilevelEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(word_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.reduce = nn.Linear(hidden_dim * 2, hidden_dim)

        conv_list = [nn.Conv2d(1, 512, (ws, hidden_dim * 2), padding=(ws - 1, 0)) for ws in [2, 3, 4]]
        self.convs = nn.ModuleList(conv_list)

    def forward(self, text, text_bow, text_lengths):
        packed = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        gru_init_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = torch.zeros(padded[0].size(0), self.hidden_dim * 2).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:text_lengths[i]], 0)
        gru_out = self.dropout(gru_out)
        gru_out = self.reduce(gru_out)

        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.leaky_relu(conv(con_out)).squeeze(3) for conv in self.convs]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        text_bow = text_bow.cuda()

        feature = torch.cat((gru_out, con_out, text_bow), 1)
        return feature
        # return gru_out
