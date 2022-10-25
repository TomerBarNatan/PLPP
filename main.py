import argparse
import dataclasses
import json
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import wandb
from dpipe.io import load
from torch.utils import data
from configs import *
from data.datasets.cc359_dataset import CC359Ds
from data. datasets.msm_dataset import MultiSiteMri
from utils.unet import UNet2D
from utils import load_model
from train_modes.pretrain import pretrain
from train_modes.pseudo_labeling import pseudo_labels_iterations


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    # lr params
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")

    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")

    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=2)

    parser.add_argument('--msm', action='store_true')
    parser.add_argument("--mode", type=str, default='pretrain', help='pretrain or PLPP')

    parser.add_argument("--target_model_path", type=str, default='')
    parser.add_argument("--pl_iterations", type=int, default=10)
    parser.add_argument("--pl_epochs", type=int, default=100)

    return parser.parse_args()


def get_configuration(args):
    if args.msm:
        if args.mode == 'pretrain':
            config = MsmPretrainConfig()
        else:
            args.source = args.target
            config = MsmConfigFinetuneClustering()

    else:
        if 'debug' in args.exp_name:
            config = DebugConfigCC359()
        elif args.mode == 'pretrain':
            config = CC359ConfigPretrain()
        else:
            config = CC359ConfigFinetuneClustering()
    return config


def get_optimizer(model, args, config):
    if config.msm:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer.zero_grad()
    return optimizer

def get_scheduler(optimizer, config):
    if config.sched:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones,
                                                         gamma=config.sched_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
    return scheduler

def move_model_to_device(model, args, config):
    if not torch.cuda.is_available():
        print('training on cpu')
        args.gpu = 'cpu'
        config.parallel_model = False
        torch.cuda.manual_seed_all(args.random_seed)

    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model

def get_datasets(args, config):
    if config.msm:
        assert args.source == args.target
        source_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/train_ids.json'))
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json'))
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/val_ids.json'), yield_id=True,
                              test=True)
        val_ds_source = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                                     test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), yield_id=True,
                               test=True)
    else:
        source_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/train_ids.json')[:config.data_len],
                            site=args.source)
        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json')[:config.data_len],
                            site=args.target)
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                         yield_id=True, slicing_interval=1)
        val_ds_source = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                                yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                          yield_id=True, slicing_interval=1)
    return source_ds, target_ds, val_ds, val_ds_source, test_ds

def init_wandb(args, config):
    project = 'adaptSegUNetMsm'
    project += f'Msm{args.mode}' if args.msm else args.mode
    if config.debug:
        wandb.init(
            project='spot3',
            id=wandb.util.generate_id(),
            name=args.exp_name,
            dir='../debug_wandb')
    else:
        wandb.init(
            project=project,
            id=wandb.util.generate_id(),
            name=args.exp_name + '_' + str(args.source) + '_' + str(args.target),
            dir='..')
def main():
    args = get_arguments()
    config = get_configuration(args)
    cudnn.enabled = True
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)

    if args.mode == "PLPP":
        config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode
        state_dict_path = Path(
            config.base_res_path) / f'source_{args.source}_target_{args.target}' / 'clustering_finetune' / 'best_model.pth'
        model = load_model(model, state_dict_path, config.msm)
    else:
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}' / args.mode
    optimizer = get_optimizer(model, args, config)
    scheduler = get_scheduler(optimizer, config)

    json.dump(dataclasses.asdict(config), open(config.exp_dir / 'config.json', 'w'))
    model.train()
    model = move_model_to_device(model, args, config)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_ds, target_ds, val_ds, val_ds_source, test_ds = get_datasets(args, config)
    init_wandb(args, config)

    trainloader = data.DataLoader(source_ds, batch_size=config.source_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)
    targetloader = data.DataLoader(target_ds, batch_size=config.target_batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)

    if args.mode == 'pretrain':
        pretrain(model, optimizer, scheduler, trainloader, config, args)
    else:
        model_path = ckpt_path
        pseudo_labels_iterations(ckpt_path, trainloader, targetloader, val_ds, test_ds, val_ds_source, args, config)


if __name__ == '__main__':
    main()
