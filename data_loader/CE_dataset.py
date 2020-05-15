import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
from utils import memcache, read_json
from PIL import Image
import hickle
import nltk


class CE(Dataset):

    def __init__(self, data_dir, text_dim, category, raw_input_dims, split, text_feat,
                 max_text_words, max_expert_tokens, transforms, vocab, attr_vocab):

        self.ordered_experts = list(raw_input_dims.keys())
        self.transforms = transforms
        self.text_dim = text_dim
        self.max_text_words = max_text_words
        self.max_expert_tokens = max_expert_tokens
        self.category = category
        self.image_dir = os.path.join(data_dir, "resized_images", f"{category}")
        self.split = split
        self.no_target = (split == 'val_trg' or split == 'test_trg')
        self.text_feat = text_feat
        self.vocab = vocab
        self.mask = False

        if self.no_target:
            self.data = read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))
            self.data_length = len(self.data)
        else:
            if text_feat == "w2v":
                text_path = f"/captions/cap.{category}.w2v.{split}.pkl"
                self.data = memcache(data_dir + text_path)
            elif text_feat == 'learnable':
                text_path = f"/captions/cap.{category}.{split}.json"
                with open(data_dir + text_path, 'r') as f:
                    self.data = json.load(f)
            else:
                raise ValueError(f"Text features {text_feat} not recognized")

            self.data_length = len(self.data)

        if "attr0" in self.ordered_experts:
            attr_path = f"/captions/asin2attr.{category}.{split.split('_')[0]}.json"
            with open(data_dir + attr_path) as f:
                self.attr = json.load(f)
            self.attr_vocab = attr_vocab
        else:
            self.attr = []

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        if self.no_target:
            candidate_id = self.data[index]
            target_id = None
            captions = None
            text = None
        elif self.split == 'test':
            candidate_id = self.data[index]['candidate']
            target_id = None
            captions = self.data[index]['captions']

            if self.text_feat == 'w2v':
                text = self.data[index]['wv']
            elif self.text_feat == 'learnable':
                text = captions
            else:
                raise ValueError
        else:
            candidate_id = self.data[index]['candidate']
            target_id = self.data[index]['target']
            captions = self.data[index]['captions']

            if self.text_feat == 'w2v':
                text = self.data[index]['wv']
            elif self.text_feat == 'learnable':
                text = captions
            else:
                raise ValueError

        # index
        candidate_ind = {expert: 1 for expert in self.ordered_experts}
        for expert in self.ordered_experts:
            if 'attr' in expert:
                if candidate_id not in self.attr:
                    candidate_ind[expert] = 0

        if target_id is not None:
            target_ind = {expert: 1 for expert in self.ordered_experts}
            for expert in self.ordered_experts:
                if 'attr' in expert:
                    if target_id not in self.attr:
                        target_ind[expert] = 0
        else:
            target_ind = None


        candidate_experts = {}
        target_experts = {}

        # resnet
        if 'resnet' in self.ordered_experts:
            candidate_fname = candidate_id + '.jpg'
            candidate_image_orig = Image.open(os.path.join(self.image_dir, candidate_fname)).convert('RGB')

            if self.mask:
                segment_fname = candidate_fname.replace('.jpg', '.hkl')
                seg = hickle.load(os.path.join(self.image_dir.replace('resized_images/', 'segmentation/'), segment_fname))
                if seg['masks'].shape[2] == 1:
                    masked_image = np.array(candidate_image_orig) * seg['masks'].astype('uint8')
                    candidate_image_orig = Image.fromarray(masked_image)
                # elif seg['masks'].shape[2] > 1:
                #     masked_image = np.array(candidate_image_orig) * np.expand_dims(seg['masks'][:, :, seg['scores'].argmax()], 2).astype('uint8')
                #     candidate_image_orig = Image.fromarray(masked_image)
                else:
                    pass
                candidate_image = self.transforms(candidate_image_orig)
            else:
                candidate_image = self.transforms(candidate_image_orig)
            candidate_experts['resnet'] = candidate_image

            if target_id is not None:
                target_fname = target_id + '.jpg'
                target_image_orig = Image.open(os.path.join(self.image_dir, target_fname)).convert('RGB')
                target_image = self.transforms(target_image_orig)
                target_experts['resnet'] = target_image

        # keypoint
        if 'keypoint' in self.ordered_experts:
            keypoint_fname = candidate_id + '.hkl'
            keypoints = hickle.load(
                os.path.join(self.image_dir.replace('resized_images/', 'backup/keypoints/'), keypoint_fname))
            cropped_images = []
            for i, keypoint in enumerate(keypoints):
                cropped_images.append(np.expand_dims(self.crop_image(candidate_image, keypoint), 0))
            candidate_experts['keypoint'] = np.concatenate(cropped_images, 0)

            if target_id is not None:
                keypoint_fname = target_id + '.hkl'
                keypoints = hickle.load(
                    os.path.join(self.image_dir.replace('resized_images/', 'backup/keypoints/'), keypoint_fname))
                cropped_images = []
                for i, keypoint in enumerate(keypoints):
                    cropped_images.append(np.expand_dims(self.crop_image(target_image, keypoint), 0))
                target_experts['keypoint'] = np.concatenate(cropped_images, 0)

        # vgg
        if 'vgg' in self.ordered_experts:
            candidate_fname = candidate_id + '.npy'
            candidate_image = np.load(
                # os.path.join(self.image_dir.replace('resized_images/', 'features/vgg_feature/'), candidate_fname))
                os.path.join(self.image_dir.replace('resized_images/', 'features/vgg/'), candidate_fname))
            candidate_experts['vgg'] = candidate_image

            if target_id is not None:
                target_fname = target_id + '.npy'
                target_image = np.load(
                    # os.path.join(self.image_dir.replace('resized_images/', 'features/vgg_feature/'), target_fname))
                    os.path.join(self.image_dir.replace('resized_images/', 'features/vgg/'), target_fname))
                target_experts['vgg'] = target_image

        # attr
        if 'attr0' in self.ordered_experts:
            if candidate_ind['attr0'] != 0:
                candidate_attr = self.attr[candidate_id]
                for i, attr in enumerate(candidate_attr):
                    if attr:
                        attr = [self.attr_vocab(token) for token in attr]
                        candidate_experts[f'attr{i}'] = np.array(attr)
                    else:
                        candidate_experts[f'attr{i}'] = np.array([np.nan])
                        candidate_ind[f'attr{i}'] = 0
            else:
                for i in range(6):
                    candidate_experts[f'attr{i}'] = np.array([np.nan])

            if target_id is not None:
                if target_ind['attr0'] != 0:
                    target_attr = self.attr[target_id]
                    for i, attr in enumerate(target_attr):
                        if attr:
                            attr = [self.attr_vocab(token) for token in attr]
                            target_experts[f'attr{i}'] = np.array(attr)
                        else:
                            target_experts[f'attr{i}'] = np.array([np.nan])
                            target_ind[f'attr{i}'] = 0
                else:
                    for i in range(6):
                        target_experts[f'attr{i}'] = np.array([np.nan])

        # get word indices for learnable word embedding
        if not self.no_target and self.text_feat == 'learnable':
            if len(text) > 1:
                tokens = nltk.tokenize.word_tokenize(str(text[0]).lower()) + ['<and>'] + \
                         nltk.tokenize.word_tokenize(str(text[1]).lower())
            else:
                tokens = nltk.tokenize.word_tokenize(str(text[0]).lower())

            text = ([self.vocab(token) for token in tokens])
            text = np.array(text)

        meta_info = {'candidate': candidate_id, 'target': target_id, 'captions': captions}

        return text, candidate_experts, target_experts, candidate_ind, target_ind, meta_info

    @staticmethod
    def crop_image(image, keypoint, m=20):
        box = np.zeros((3, m * 2, m * 2))
        x1 = max(0, keypoint[0] - m)
        x2 = min(244, keypoint[0] + m)
        y1 = max(0, keypoint[1] - m)
        y2 = min(244, keypoint[1] + m)
        cropped_image = image[:, x1:x2, y1:y2]
        x = cropped_image.size(1)
        y = cropped_image.size(2)
        box[:, :x, :y] = cropped_image
        return box

    def collate_fn(self, batch):
        text_feature, candidate, target, candidate_ind, target_ind, meta_info = zip(*batch)

        candidate_ind = {expert: [x[expert] for x in candidate_ind] for expert in self.ordered_experts}
        candidate_ind = {key: torch.Tensor(val) for key, val in candidate_ind.items()}

        if not self.no_target and not self.split == 'test':
            target_ind = {expert: [x[expert] for x in target_ind] for expert in self.ordered_experts}
            target_ind = {key: torch.Tensor(val) for key, val in target_ind.items()}

        candidate_experts = []
        target_experts = []

        # text
        if self.no_target:
            text = None
            text_lengths = None
            text_bow = None
            text_mean = None
        else:
            batch_size = len(text_feature)
            text_lengths = [len(x) for x in text_feature]
            max_text_words = min(max(text_lengths), self.max_text_words)

            if self.text_feat == "w2v":
                text = np.zeros((batch_size, max_text_words, self.text_dim))
                for i, feature in enumerate(text_feature):
                    text[i, :text_lengths[i], :] = feature
                text = torch.FloatTensor(text)
                text_bow = None
            elif self.text_feat == "learnable":
                text = np.zeros((batch_size, max_text_words))
                text_bow = np.zeros((batch_size, len(self.vocab)))
                for i, feature in enumerate(text_feature):
                    text[i, :text_lengths[i]] = feature
                    text_bow[i] = np.bincount(feature, minlength=len(self.vocab))
                text = torch.LongTensor(text)
                text_bow = torch.FloatTensor(text_bow)
            else:
                raise ValueError

            text_mean = torch.cat([torch.Tensor(np.expand_dims(t.mean(0), 0)) for t in text_feature], 0)

        # experts
        for expert in self.ordered_experts:
            if expert in {'resnet', 'keypoint'}:
                candidate_val = np.vstack([np.expand_dims(x[expert], 0) for x in candidate])
            elif 'attr' in expert:
                attr_lengths = [len(x[expert]) for x in candidate]
                max_attr_tokens = min(max(attr_lengths), self.max_expert_tokens)
                candidate_val = np.zeros((len(attr_lengths), max_attr_tokens))
                for i, feature in enumerate(candidate):
                    if np.isnan(feature[expert])[0]:
                        candidate_val[i, :] = feature[expert].repeat(max_attr_tokens)
                    else:
                        candidate_val[i, :attr_lengths[i]] = feature[expert]
            else:
                candidate_val = np.vstack([x[expert] for x in candidate])

            candidate_experts.append((expert, torch.from_numpy(candidate_val)))

        candidate_experts = OrderedDict(candidate_experts)

        if not self.no_target and not self.split == 'test':
            for expert in self.ordered_experts:
                if expert in {'resnet', 'keypoint'}:
                    target_val = np.vstack([np.expand_dims(x[expert], 0) for x in target])
                elif 'attr' in expert:
                    attr_lengths = [len(x[expert]) for x in target]
                    max_attr_tokens = min(max(attr_lengths), self.max_expert_tokens)
                    target_val = np.zeros((len(attr_lengths), max_attr_tokens))
                    for i, feature in enumerate(target):
                        if np.isnan(feature[expert])[0]:
                            target_val[i, :] = feature[expert].repeat(max_attr_tokens)
                        else:
                            target_val[i, :attr_lengths[i]] = feature[expert]
                else:
                    target_val = np.vstack([x[expert] for x in target])

                target_experts.append((expert, torch.from_numpy(target_val)))

            target_experts = OrderedDict(target_experts)

        return {'text': text,
                'text_bow': text_bow,
                'text_lengths': text_lengths,
                'text_mean': text_mean,
                'candidate_experts': candidate_experts,
                'target_experts': target_experts,
                'candidate_ind': candidate_ind,
                'target_ind': target_ind,
                'meta_info': meta_info}
