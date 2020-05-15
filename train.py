import os
import argparse
import time
import random
import numpy as np
import torch
from parse_config import ConfigParser
from utils import compute_dims
import data_loader.data_loaders as module_data
import model.model as module_arch
import model.loss as module_loss
from trainer import Trainer
from utils.vocab import Vocabulary
from utils.text2vec import get_we_parameter
from test import test
from multiprocessing import set_start_method


def main(config):
    logger = config.get_logger('train')
    expert_dims, raw_input_dims = compute_dims(config)
    seeds = [int(x) for x in config._args.seeds.split(',')]

    for seed in seeds:
        tic = time.time()
        logger.info(f"Setting experiment random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if config['experts']['text_feat'] == 'learnable':
            # vocab
            vocab = Vocabulary()
            vocab.load('dataset/captions/dict.all_200k_gan.json')
            vocab_size = len(vocab)
            if config['experts']['text_feat_init'] == True:
                # word2vec, download file and move to we_root-path directory
                # https://www.kaggle.com/jacksoncrow/word2vec-flickr30k/version/1
                we_rootpath = '/home/yj/pretrained_model'
                w2v_data_path = os.path.join(we_rootpath, "word2vec/", 'flickr', 'vec500flickr30m')
                we_parameter = get_we_parameter(vocab, w2v_data_path)
            else:
                we_parameter = None
        else:
            vocab = None
            vocab_size = None
            we_parameter = None

        if "attr" in config['experts']['modalities']:
            attr_vocab = Vocabulary()
            attr_vocab.load('dataset/captions/dict.attr.json')
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
            text_feat=config['experts']['text_feat'],
        )
        # logger.info(model)

        loss = config.init(name='loss', module=module_loss)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(
            model,
            loss,
            optimizer,
            config=config,
            data_loaders=data_loaders,
            lr_scheduler=lr_scheduler,
        )

        trainer.train()
        best_ckpt_path = config.save_dir / "trained_model.pth"
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")

        test_args = argparse.ArgumentParser()
        test_args.add_argument("--device", default=config._args.device)
        test_args.add_argument("--resume", default=best_ckpt_path)
        test_config = ConfigParser(test_args)
        test(test_config)


if __name__ == '__main__':
    # for vscode debugging
    set_start_method('spawn', True)

    args = argparse.ArgumentParser()
    args.add_argument('--config', default='configs/ce/train.json', type=str)
    args.add_argument('--device', default=None, type=str)
    args.add_argument('--resume', default=None, type=str)
    args.add_argument('--seeds', default="0", type=str)
    args = ConfigParser(args)

    print("Launching experiment with config:")
    print(args)
    main(args)
