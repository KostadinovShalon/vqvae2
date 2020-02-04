import visdom
from matplotlib import pyplot as plt
import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    acc = 0.0
    n = 0
    total_correct = 0
    total_images = 0

    for i, (top, bottom, label) in enumerate(loader):
        n += 1
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        total_loss += loss.item()
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        total_correct += correct.sum()
        total_images += target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )
    return total_loss / n, total_correct / total_images


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        ar = np.array(x)

        return torch.from_numpy(ar).long()


def plot(loss, acc, title, vis, win=None):
    f = plt.figure(figsize=(16, 8))
    f.suptitle(title)
    ax = f.add_subplot(1, 2, 1)
    ax.plot(loss)
    ax.set_title('Loss.')
    ax.set_xlabel('Epoch')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(acc)
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')

    if win is None:
        win = vis.matplot(f)
    else:
        vis.matplot(f, win=win)

    plt.close(f)

    return win


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--vishost', type=str, default='localhost')
    parser.add_argument('--visport', type=int, default=48048)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    if args.hier == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    vis = visdom.Visdom(server=args.vishost, port=args.visport)
    win = None
    losses = []
    accs = []
    for i in range(args.epoch):
        l, a = train(args, i, loader, model, optimizer, scheduler, device)
        losses.append(l)
        accs.append(a)
        win = plot(losses, accs, args.hier, vis, win)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoints/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )
