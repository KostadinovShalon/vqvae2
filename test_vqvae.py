from tqdm import tqdm
from torch import nn
import torch
import torchvision.utils


def test(epoch, loader, model, device):
    model.eval()
    loader = tqdm(loader)

    criterion = nn.MSELoss()  # Reconstruction criterion

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0
    loss = 0
    commitment_loss = 0

    for i, (x, label) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            out, cmt_loss = model(x)
            recon_loss = criterion(out, x)
            cmt_loss = cmt_loss.mean()
            loss = recon_loss + latent_loss_weight * cmt_loss

        mse_sum += recon_loss.item() * x.shape[0]
        commitment_loss += cmt_loss.item() * x.shape[0]
        loss = loss.item() * x.shape[0]
        mse_n += x.shape[0]

        description = (
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'commitment_loss: {cmt_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}'
            )
        )

    return loss, mse_sum, commitment_loss


def recon_sample(epoch, model, loader, device, sample_size=8):
    model.eval()

    samples = []

    for x, label in loader:
        bs = x.shape[0]
        if bs > sample_size:
            samples.append(x[:sample_size])
            break
        else:
            remaining = sample_size - len(samples) * bs
            if remaining > bs:
                samples.append(x)
            else:
                samples.append(x[:remaining])
                break

    sample = torch.cat(samples)
    sample = sample.to(device)
    with torch.no_grad():
        out, _ = model(sample)

    torchvision.utils.save_image(
        torch.cat([sample, out], 0),
        f'sample/recon_epoch_{str(epoch + 1).zfill(5)}.png',
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )
