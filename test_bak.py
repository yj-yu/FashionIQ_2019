import os
import argparse
from pathlib import Path
import time
import torch
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
from model.metric import sharded_cross_view_inner_product
from utils.util import compute_dims
from utils.vocab import Vocabulary
from utils.text2vec import get_we_parameter


def test(config):
    config.config['data_loader']['args']['mode'] = 'test'
    logger = config.get_logger('test')
    logger.info("Running test with configuration:")
    logger.info(config)

    expert_dims, raw_input_dims = compute_dims(config)

    if config['experts']['text_feat'] == 'learnable':
        # vocab
        vocab = Vocabulary()
        vocab.load('/data1/common_datasets/fashion_data/captions/dict.all_200k_gan.json')
        vocab_size = len(vocab)

        # word2vec
        we_rootpath = '/home/yj/fashion-ret/start_kit/pretrained_model'
        w2v_data_path = os.path.join(we_rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        we_parameter = get_we_parameter(vocab, w2v_data_path)
    else:
        vocab = None
        vocab_size = None
        we_parameter = None

    if "attr" in config['experts']['modalities']:
        attr_vocab = Vocabulary()
        attr_vocab.load('/data1/common_datasets/fashion_data/captions/dict.attr.json')
        attr_vocab_size = len(attr_vocab)
    else:
        attr_vocab = None
        attr_vocab_size = None

    data_loaders = config.init(
        name='data_loader',
        module=module_data,
        raw_input_dims=raw_input_dims,
        text_feat=config['experts']['text_feat'],
        text_dim=config['experts']['text_dim'],
        vocab=vocab,
        attr_vocab=attr_vocab,
        pretrain=config['trainer']['pretrain']
    )

    model = config.init(
        name='arch',
        module=module_arch,
        expert_dims=expert_dims,
        text_dim=config['experts']['text_dim'],
        same_dim=config['experts']['ce_shared_dim'],
        we_parameter=we_parameter,
        vocab_size=vocab_size,
        attr_vocab_size=attr_vocab_size,
        text_feat=config['experts']['text_feat']
    )

    ckpt_path = Path(config._args.resume)
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running test on {device}")

    model = model.to(device)
    model.eval()

    categories = ['dress', 'shirt', 'toptee']
    modalities = data_loaders[categories[0]].dataset.ordered_experts
    metric = {'score': dict()}

    for i, category in enumerate(categories):
        val_experts = {expert: list() for expert in modalities}
        data_asin = []

        # Target
        for batch in data_loaders[category + '_trg']:
            for key, val in batch['candidate_experts'].items():
                batch['candidate_experts'][key] = val.to(device)

            data_asin.extend([meta['candidate'] for meta in batch['meta_info']])

            with torch.no_grad():
                experts, _ = model(batch['candidate_experts'], batch['ind'])
                for modality, val in experts.items():
                    val_experts[modality].append(val)

        for modality, val in val_experts.items():
            val_experts[modality] = torch.cat(val)

        scores = []
        meta_infos = []
        # Source
        for batch in data_loaders[category]:
            for experts in ['candidate_experts']:
                for key, val in batch[experts].items():
                    batch[experts][key] = val.to(device)
            batch["text"] = batch["text"].to(device)

            meta_infos.extend(list(batch['meta_info']))

            with torch.no_grad():
                composition_feature, moe_weights = model(batch['candidate_experts'], batch['ind'], batch['text'], batch['text_lengths'])

                cross_view_conf_matrix = sharded_cross_view_inner_product(
                    vid_embds=val_experts,
                    text_embds=composition_feature,
                    text_weights=moe_weights,
                    subspaces=model.image_encoder.modalities,
                    l2renorm=True,
                    dist=True
                )
                scores.append(cross_view_conf_matrix)
        scores = torch.cat(scores)
        val_ids = data_loaders[category + '_trg'].dataset.data
        assert val_ids == data_asin
        metric['score'][category] = {'ids': val_ids, 'matrix': scores, 'meta_info': meta_infos}

    save_fname = ckpt_path.parent / f'test_score.pt'
    tic = time.time()
    logger.info("Saving score matrix: {} ...".format(save_fname))
    torch.save(metric, save_fname)
    logger.info(f"Done in {time.time() - tic:.3f}s")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='configs/ce/test.json', type=str, help='config file path')
    args.add_argument('--resume', default=None, type=str, help='path to checkpoint for test')
    args.add_argument('--device', default='0', type=str, help='indices of GPUs to enable')
    test_config = ConfigParser(args)

    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert test_config._args.resume, msg

    test(test_config)
