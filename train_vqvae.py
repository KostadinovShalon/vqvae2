import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from tqdm import tqdm

from scheduler import CycleScheduler
from test_vqvae import test, recon_sample
from vqvae import VQVAE

import os.path

import visdom
import matplotlib.pyplot as plt

import numpy as np


def train(epoch, loader, model, optimizer, scheduler, device):
    model.train()
    loader = tqdm(loader)

    criterion = nn.MSELoss()  # Reconstruction criterion

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0
    total_loss = 0

    for i, (x, label) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()

        out, commitment_loss = model(x)
        recon_loss = criterion(out, x)
        commitment_loss = commitment_loss.mean()
        loss = recon_loss + latent_loss_weight * commitment_loss
        loss.backward()

        optimizer.step()

        mse_sum += recon_loss.item() * x.shape[0]
        mse_n += x.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'commitment_loss: {commitment_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        total_loss += loss.item() * x.shape[0]

    if scheduler is not None:
        scheduler.step()

    return total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', help='Image size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--sched', type=str, default='cycle')
    parser.add_argument('--vishost', type=str, default='localhost')
    parser.add_argument('--visport', type=int, default=8097)
    parser.add_argument('path', help="root path with train and test folder in it", type=str)

    args = parser.parse_args()

    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    )

    train_path = os.path.join(args.path, "train")
    test_path = os.path.join(args.path, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(train_loader) * args.epoch, momentum=None
        )
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 70], 0.1)

    train_losses = []
    test_losses = []
    vis = visdom.Visdom(server=args.vishost, port=args.visport)
    win = None
    best_model_loss = np.inf
    for i in range(args.epoch):
        # Training stage
        print(f"Training epoch {i + 1}")
        train_loss = train(i, train_loader, model, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.5f}")

        # Testing stage
        print(f"Testing epoch {i + 1}")
        test_loss, test_recon_error, test_commitment_loss = test(i, test_loader, model, device)
        print(f"Test Loss: {test_loss:.5f}")
        torch.save(
            model.state_dict(), f'checkpoints/vqvae_chkpt_{str(i + 1).zfill(3)}.pt'
        )

        if test_loss < best_model_loss:
            print("Saving model")
            torch.save(model.state_dict(), f'weights/vqvae.pt')
            best_model_loss = test_loss

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        win = plot(train_losses, test_losses, vis, win)

        # Sampling stage
        recon_sample(i, model, test_loader, device)


def plot(train_losses, test_losses, vis, win=None):
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_losses)
    ax.set_yscale('log')
    ax.set_title('Train Loss.')
    ax.set_xlabel('Epoch')

    ax = f.add_subplot(1, 2, 2)
    ax.plot(test_losses)
    ax.set_yscale('log')
    ax.set_title('Test Loss')
    ax.set_xlabel('Epoch')

    if win is None:
        win = vis.matplot(f)
    else:
        vis.matplot(f, win=win)

    plt.close(f)

    return win


if __name__ == '__main__':
    main()
