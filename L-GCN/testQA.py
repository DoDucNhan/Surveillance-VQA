import logging
from logging import Logger
import math
import os
import pickle
import random
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
# import ipdb
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory, ConfigTree
from termcolor import colored
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from arguments import args
from model import get_model
from utils import MULTIPLE_CHOICE_TASKS, accuracy, count_correct
from utils.config import config
from utils.logging import set_default_logger
from utils.meters import AverageMeter
from dataset import collate_fn, snn_dataset


logger = logging.getLogger(__name__)

def get_dataloader(opt: ConfigTree, logger: Logger) -> (DataLoader, DataLoader):
    test_set = snn_dataset.get_dataset(opt.get_config('dataset'), logger, args.question_type)

    test_loader = DataLoader(
        test_set,
        batch_size=opt.get_int('batch_size'),
        shuffle=False,
        num_workers=0 if args.debug else opt.get_int('num_workers'),
        pin_memory=True,
        collate_fn=collate_fn
    )

    return test_loader

@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, criterion: nn.Module, qType: str) -> float:
    loader_length = len(loader)

    losses = AverageMeter('Loss')

    if TASK in utils.MULTIPLE_CHOICE_TASKS or TASK in ['frameqa', 'youtube2text']:
        result = AverageMeter('Acc')
    else:
        result = AverageMeter('MSE')

    type_meters = dict()

    model.eval()

    final_out = []

    for i, data in enumerate(tqdm(loader)):

        data = utils.batch_to_gpu(data)

        (
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox,
            answer
        ) = data

        if config.get_bool('abc.is_multiple_choice'):
            answer = torch.zeros_like(answer)

        out = model(
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox
        )

        loss: torch.Tensor = criterion(out, answer)

        compute_score(losses, result, out, answer, loss)
        
        print(out)
        print(answer)


    writer.add_scalar(f'Test/{losses.name}', losses.avg, 1)
    writer.add_scalar(f'Test/{result.name}', result.avg, 1)

    return result.avg, type_meters


@torch.no_grad()
def compute_score(losses: AverageMeter, result: AverageMeter, out: torch.Tensor, answer: torch.Tensor,
                  loss: torch.Tensor):
    batch_size = answer.shape[0]

    if TASK in utils.MULTIPLE_CHOICE_TASKS or TASK in ['frameqa', 'youtube2text']:
        acc = accuracy(out, answer)
        result.update(acc.item(), batch_size)
    elif TASK == 'count':
        out = out * 10. + 1.
        mse = F.mse_loss(out.round().clamp(1., 10.), answer.clamp(1., 10.))
        result.update(mse.item(), batch_size)

    if TASK in MULTIPLE_CHOICE_TASKS or config.get_bool('abc.is_multiple_choice'):
        losses.update(loss.item() / batch_size, batch_size)
    else:
        losses.update(loss.item(), batch_size)


def main():
    test_loader = get_dataloader(config, logger)

    num_classes = 1
    if TASK == 'frameqa':
        answer_dict = utils.load_answer_dict()
        num_classes = len(answer_dict)

    logger.info(f'Num classes: {num_classes}')

    vocab_size = utils.get_vocab_size(config, TASK, level='word')
    char_vocab_size = utils.get_vocab_size(config, TASK, level='char')

    model = get_model(vocab_size, char_vocab_size, num_classes)
    model = model.cuda()

    if TASK in MULTIPLE_CHOICE_TASKS:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif TASK == 'count':
        inner_criterion = nn.MSELoss()

        def criterion(input, target):
            target = (target - 1.) / 10.
            return inner_criterion(input, target)

        # criterion = nn.SmoothL1Loss()
    elif TASK in ['frameqa']:
        criterion = nn.CrossEntropyLoss()

    elif TASK == 'youtube2text':
        if config.get_bool('abc.is_multiple_choice'):
            criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer_type = config.get_string('optimizer')

    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.get_float('adam.lr'))
    else:
        raise Exception(f'Unknow optimizer: {optimizer_type}')

    checkpoint = torch.load(os.path.join(args.experiment_path, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    result, type_meters = test(model, test_loader, criterion, args.question_type)

    if TASK == 'count':
        logger.info(f'Best MSE: {result}')
    else:
        logger.info(f'Best Acc on {args.question_type}: {result}')


def fix_seed(config):
    seed = config.get_int('seed')
    logger.info(f'Set seed={seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    
    set_default_logger(args.experiment_path, debug=False)
    # config = ConfigFactory.parse_file(args.config)

    fix_seed(config)

    pprint(config)

    TASK = config.get_string('task')

    best_meters = dict()

    if args.experiment_path is not None:
        writer = SummaryWriter(log_dir=args.experiment_path)
    else:
        # writer: SummaryWriter = FakeObj()
        raise Exception('No exp path for tensorboard')

    main()

    writer.close()
