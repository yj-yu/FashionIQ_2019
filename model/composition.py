import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion import get_fusion


class TIRG(nn.Module):
    def __init__(self, fusion, embed_dim=512):
        super(TIRG, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        if fusion == 'base':
            concat_num = 2
        else:
            concat_num = 3

        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

        if fusion == 'hadamard':
            self.fusion = get_fusion(fusion)
        elif fusion == 'base':
            self.fusion = None
        else:
            self.fusion = get_fusion(fusion, embed_dim, embed_dim, embed_dim, None, None, 0.2)

    def forward(self, imgs, texts):
        if len(texts.size()) > 2:
            texts = texts.squeeze(1)
        if self.fusion is None:
            x = torch.cat([imgs, texts], dim=1)
        else:
            fusion = self.fusion(imgs, texts)
            x = torch.cat([imgs, texts, fusion], dim=1)
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = F.sigmoid(f1) * imgs * self.a[0] + f2 * self.a[1]
        return f


class TrgTIRG(nn.Module):
    def __init__(self, fusion='base', embed_dim=512):
        super(TrgTIRG, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        if fusion == 'base':
            concat_num = 2
        else:
            concat_num = 3

        self.gated_feature_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(concat_num * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(concat_num * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

        if fusion == 'hadamard':
            self.fusion = get_fusion(fusion)
        elif fusion == 'base':
            self.fusion = None
        else:
            self.fusion = get_fusion(fusion, embed_dim, embed_dim, embed_dim, None, None, 0.2)

    def forward(self, imgs, texts):
        if len(texts.size()) > 2:
            texts = texts.squeeze(1)
        if self.fusion is None:
            x = torch.cat([imgs, texts], dim=1)
        else:
            fusion = self.fusion(imgs, texts)
            x = torch.cat([imgs, texts, fusion], dim=1)
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = F.sigmoid(f1) * imgs * self.a[0] + f2 * self.a[1] + 10.0 * imgs
        return f


class NormalizationLayer(nn.Module):
    def __init__(self, normalize_scale=5.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, imgs, texts):
        if len(texts.size()) > 2:
            texts = texts.squeeze(1)
        f = imgs + texts
        return f


class AttrMLB(nn.Module):
    def __init__(self, embed_dim=1024):
        super(AttrMLB, self).__init__()

        self.attention_net1 = nn.Sequential(
            nn.Linear(9600, embed_dim),
            nn.Tanh())
        self.attention_net2 = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.Tanh())
        self.attention_net3 = nn.Linear(embed_dim, 1)

        self.output_net1 = nn.Sequential(
            nn.Linear(9600, embed_dim),
            nn.Tanh())
        self.output_net2 = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.Tanh())

    def forward(self, image, attr):
        attr_attention_input = self.attention_net1(attr).unsqueeze(1).expand(-1, 49, -1)
        image = image.flatten(2, -1).transpose(2, 1)
        img_attention_input = self.attention_net2(image)
        attention = self.attention_net3(attr_attention_input * img_attention_input).softmax(1)
        attended_image = (attention * image).mean(1)
        output = self.output_net1(attr) * self.output_net2(attended_image)

        return output
