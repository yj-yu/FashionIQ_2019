import sys
import collections
import time
import torch
import numpy as np
from base import BaseTrainer
from model.metric import sharded_cross_view_inner_product
from model.metric import compute_score


class Trainer(BaseTrainer):

    def __init__(self, model, loss, optimizer, config, data_loaders, lr_scheduler):
        super().__init__(model, loss, optimizer, config)
        self.config = config
        self.data_loaders = data_loaders
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.data_loaders["train"])
        self.log_step = int(np.sqrt(len(data_loaders["train"].dataset)))

    def _train_epoch(self, epoch, mode="train"):
        self.model.train()
        total_loss = 0
        progbar = Progbar(len(self.data_loaders[mode].dataset))
        modalities = self.model.image_encoder.modalities

        for batch_idx, batch in enumerate(self.data_loaders[mode]):
            for experts in ['candidate_experts', 'target_experts']:
                for key, val in batch[experts].items():
                    batch[experts][key] = val.to(self.device)
            batch["text"] = batch["text"].to(self.device)

            batch['text_mean'] = batch['text_mean'].to(self.device)

            self.optimizer.zero_grad()

            src_experts = self.model.image_encoder(batch['candidate_experts'], batch['candidate_ind'])
            trg_experts = self.model.image_encoder(batch['target_experts'], batch['target_ind'])

            # diff = {}
            # for mod in modalities:
            #     diff[mod] = src_experts[mod] - trg_experts[mod]

            src_text, moe_weights = self.model.get_text_feature(batch['text'],
                                                                batch['candidate_ind'],
                                                                batch['text_bow'],
                                                                batch['text_lengths'])

            trg_text, _ = self.model.get_text_feature(batch['text'],
                                                      batch['target_ind'],
                                                      batch['text_bow'],
                                                      batch['text_lengths'],
                                                      target=True)
            # trg_text, _ = self.model.text_encoder['trg'](batch['text_mean'].unsqueeze(1), batch['target_ind'])

            src_feature = self.model.get_combined_feature(src_experts, src_text)
            trg_feature = self.model.get_combined_feature(trg_experts, trg_text, target=True)

            # trg_feature = {}
            # for key in trg_experts.keys():
            #     trg_feature[key] = self.model.trg_normalization_layer(trg_experts[key] + 0.1 * trg_text[key].squeeze(1)).unsqueeze(1)

            cross_view_conf_matrix = sharded_cross_view_inner_product(
                vid_embds=trg_feature,
                text_embds=src_feature,
                text_weights=moe_weights,
                subspaces=self.model.image_encoder.modalities,
            )

            # for key, val in trg_experts.items():
            #     trg_experts[key] = val.unsqueeze(1)
             
            # aux_loss = sharded_cross_view_inner_product(
            #     vid_embds=trg_experts,
            #     text_embds=src_feature,
            #     text_weights=moe_weights,
            #     subspaces=self.model.image_encoder.modalities,
            # )
            # target_feature, _, _ = self.model(batch['target_experts'],
            #                                   batch['target_ind'],
            #                                   batch['text'],
            #                                   batch['text_bow'],
            #                                   batch['text_lengths'],
            #                                   target=True)
            #
            # combined_feature, _, moe_weights = self.model(batch['candidate_experts'],
            #                                               batch['candidate_ind'],
            #                                               batch['text'],
            #                                               batch['text_bow'],
            #                                               batch['text_lengths'])

            # combined_target_feature = dict()
            # for i, mod in enumerate(modalities):
            #     combined_target_feature[mod] = \
            #         self.model.normalization_layer(
            #             self.model.target_composition[i](target_feature[mod], text_feature[mod])).unsqueeze(1)

            # cross_view_conf_matrix = sharded_cross_view_inner_product(
            #     vid_embds=target_feature,
            #     text_embds=combined_feature,
            #     text_weights=moe_weights,
            #     subspaces=self.model.image_encoder.modalities,
            # )

            # loss1 = self.loss(cross_view_conf_matrix)
            # loss2 = self.loss(aux_loss)
            # loss = loss1 + loss2
            loss = self.loss(cross_view_conf_matrix)
            loss.backward()
            self.optimizer.step()

            batch_size = len(batch['text'])
            progbar.add(batch_size, values=[('loss', loss.item())])
            total_loss += loss.item()

        if mode == "train":
            log = {'loss': total_loss / self.len_epoch}

            if epoch > self.val_epoch:
                val_log = self._valid_epoch()
                log.update(val_log)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            log = None

        return log

    def _valid_epoch(self):
        self.model.eval()
        categories = self.config['data_loader']['args']['categories']
        metric = {'recall': np.zeros((3, 2)), 'score': dict(), 'largest': self.largest}
        modalities = self.data_loaders[categories[0]].dataset.ordered_experts

        for i, category in enumerate(categories):
            val_experts = {expert: list() for expert in modalities}
            target_ind = {expert: list() for expert in modalities}

            for batch in self.data_loaders[category + '_trg']:
                for key, val in batch['candidate_experts'].items():
                    batch['candidate_experts'][key] = val.to(self.device)

                for key, val in batch['candidate_ind'].items():
                    target_ind[key].append(val)

                with torch.no_grad():
                    experts = self.model.image_encoder(batch['candidate_experts'], batch['candidate_ind'])
                    # experts, _, _ = self.model(batch['candidate_experts'], batch['candidate_ind'], target=True)
                    for modality, val in experts.items():
                        val_experts[modality].append(val)

            for modality, val in val_experts.items():
                val_experts[modality] = torch.cat(val)

            for modality, val in target_ind.items():
                target_ind[modality] = torch.cat(val)

            scores = []
            meta_infos = []
            val_size = val_experts['resnet'].size(0)

            for batch in self.data_loaders[category]:
                for experts in ['candidate_experts']:
                    for key, val in batch[experts].items():
                        batch[experts][key] = val.to(self.device)
                batch["text"] = batch["text"].to(self.device)
                batch_size = batch["text"].size(0)
                batch['text_mean'] = batch['text_mean'].to(self.device)

                meta_infos.extend(list(batch['meta_info']))

                with torch.no_grad():
                    src_experts = self.model.image_encoder(batch['candidate_experts'], batch['candidate_ind'])
                    src_text, moe_weights = self.model.get_text_feature(batch['text'],
                                                                        batch['candidate_ind'],
                                                                        batch['text_bow'],
                                                                        batch['text_lengths'])
                    src_feature = self.model.get_combined_feature(src_experts, src_text)

                    trg_text, _ = self.model.get_text_feature(batch['text'],
                                                              batch['target_ind'],
                                                              batch['text_bow'],
                                                              batch['text_lengths'],
                                                              target=True)
                    # trg_text, _ = self.model.text_encoder['trg'](batch['text_mean'].unsqueeze(1), batch['target_ind'])

                    batch_target = dict()
                    for h, mod in enumerate(modalities):
                        tmp = []
                        for k in range(batch_size):
                            # diff = src_experts[mod][k].expand(val_size, -1) - val_experts[mod]
                            # tmp.append(self.model.normalization_layer(self.model.target_composition[h](val_experts[mod], trg_text[mod][k].expand(val_size, -1))))
                            tmp.append(self.model.trg_normalization_layer(self.model.target_composition[h](val_experts[mod], trg_text[mod][k].expand(val_size, -1))))
                            # tmp.append(self.model.trg_normalization_layer(val_experts[mod] + 0.1 * trg_text[mod][k].expand(val_size, -1)))
                            # tmp.append(self.model.trg_normalization_layer(self.model.target_composition[h](val_experts[mod], diff)))
                        batch_target[mod] = torch.stack(tmp)

                    cross_view_conf_matrix = sharded_cross_view_inner_product(
                        vid_embds=batch_target,
                        text_embds=src_feature,
                        text_weights=moe_weights,
                        subspaces=self.model.image_encoder.modalities,
                        l2renorm=True,
                        dist=True,
                        val=True
                    )
                    scores.append(cross_view_conf_matrix)

            scores = torch.cat(scores)
            val_ids = self.data_loaders[category + '_trg'].dataset.data

            for j, score in enumerate(scores):
                _, topk = score.topk(dim=0, k=50, largest=self.largest, sorted=True)
                meta_infos[j]['ranking'] = [val_ids[idx] for idx in topk]

            r10, r50 = compute_score(meta_infos, meta_infos)
            metric['recall'][i] = r10, r50
            metric['score'][category] = {'ids': val_ids, 'matrix': scores, 'meta_info': meta_infos}

        metric['recall_avg'] = metric['recall'].mean()

        return metric


class Progbar:
    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
