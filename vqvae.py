from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class Quantize(nn.Module):
    """
        Vector Quantization module that performs codebook look up
    """

    def __init__(self, embedding_dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5):
        """
        Parameters
        ----------
        :param embedding_dim: code dimension
        :param n_embed: number of embeddings per latent space
        :param decay:  decay value for codeblock exponential moving average update
        :param eps: epsilon value to avoid division by 0
        """
        super().__init__()

        assert 0 <= decay <= 1

        self.dim = embedding_dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embedding_dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z: torch.Tensor):
        """
        :param z: Encoder output
        :return: Quantized tensor
        """
        flatten = z.reshape(-1, self.dim)  # Converting the z input to a [N x D] tensor, where D is embedding dimension
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # Distance calculation between Ze and codebook.
        _, embed_ind = (-dist).max(1)  # Arg min of closest distances
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # Assigns the actual codes according
        # their closest indices, with flattened
        embed_ind = embed_ind.view(*z.shape[:-1])  # B x C x H x W tensor with the indices of their nearest code
        quantize = self.embed_code(embed_ind)  # B x C x H x W x D quantized tensor

        # Exponential decay updating, as a replacement to codebook loss
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Commitment loss, used to keep the encoder output close to the codebook
        diff = (quantize.detach() - z).pow(2).mean()

        quantize = z + (quantize - z).detach()  # This is added to pass the gradients directly from Z. Basically
        # means that quantization operations have no gradient

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """
        Looks up for the codes in self.embed using embed_id
        :param embed_id: indices to search in the embedding
        :return: a ( embed_id.shape() X embedding_dim) tensor with neartes codebook embeddings
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    """
        Residual block with two Convolutional layers
    """
    def __init__(self, in_channel, channel):
        """
        :param in_channel: input channels
        :param channel: intermediate channels of residual block
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, inp):
        out = self.conv(inp)
        out += inp

        return out


class Encoder(nn.Module):
    """
    Encoder network. It is based on a set of convolutional layers followed by N residual blocks.
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        """
        :param in_channel: input channels
        :param channel: output channels
        :param n_res_block: number of residual blocks
        :param n_res_channel: number of intermediate layers of the residual block
        :param stride: stride to reduce the input image dimensions (it can be 2 or 4)
        """
        super().__init__()
        assert stride == 2 or stride == 4

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        else:  # stride = 2
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    """
        Decoder network. It consists on a convolutional layer, N residual blocks and a set of deconvolutions.
    """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        """
                :param in_channel: input channels
                :param channel: output channels
                :param n_res_block: number of residual blocks
                :param n_res_channel: number of intermediate layers of the residual block
                :param stride: stride to reduce the input image dimensions (it can be 2 or 4)
        """
        super().__init__()

        assert stride == 2 or stride == 4

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        else:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        return self.blocks(z)


class VQVAE(nn.Module):
    """
     Vector Quantized Variational Autoencoder. This networks includes a encoder which maps an
     input image to a discrete latent space, and a decoder to maps the latent map back to the input domain
    """
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        """
        :param in_channel: input channels
        :param channel: output channels of the encoder
        :param n_res_block: number of residual blocks for the decoder and the encoder
        :param n_res_channel: number of intermediate channels of the residual block
        :param embed_dim: embedding dimensions
        :param n_embed: number of embeddings in the codebook
        :param decay: weight decay for exponential updating
        """
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)  # Bottom encoder
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)  # Top encoder
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)  # Dimension reduction to embedding size
        self.quantize_t = Quantize(embed_dim, n_embed, decay)  # Top vector quantization
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )  # Top decoder
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)  # Bottom vector quantization dimension red
        self.quantize_b = Quantize(embed_dim, n_embed)  # Bottom vector quantization
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )  # Top upsampling to bottom channels
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, x):
        quant_t, quant_b, diff, _, _ = self.encode(x)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes and quantizes an input tensor using VQ-VAE2 algorithm
        :param x: input tensor
        :return: A tuple containing: quantized top map, quantized bottom map, commitment loss of top and bottom
                 maps, codebook indices used for top map and codebook indices for bottom map.
        """
        enc_b = self.enc_b(x)  # Encoding bottom
        enc_t = self.enc_t(enc_b)  # Encoding top from bottom encoding

        # Quantization of top layer and converting to B x H x W x C
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)

        # converting back the quantized map to BxCxHxW
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)  # Commitment loss of top layer

        dec_t = self.dec_t(quant_t)  # Decoding top quantized map "one level" to concatenate it with bottom encoding
        enc_b = torch.cat([dec_t, enc_b], 1)  # Concatenation of Ebottom(x) and decoded top quantized map

        # Quantization of bottom encoding and decoded top map
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)  # commitment loss of bottom layer

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t: torch.Tensor, quant_b: torch.Tensor) -> torch.Tensor:
        """
        Decodes top and bottom latent mappings
        :param quant_t: quantized top codes
        :param quant_b: quantized bottom codes
        :return: decoded tensor in input space
        """
        upsample_t = self.upsample_t(quant_t)  # Upsamples top quantization
        quant = torch.cat([upsample_t, quant_b], 1)  # Concatenates bottom and upsampled top latent maps
        dec = self.dec(quant)  # Decodes to input space

        return dec

    def decode_code(self, code_t: torch.Tensor, code_b: torch.Tensor) -> torch.Tensor:
        """
        Decodes top and bottom latent mappings given the code indices
        :param code_t: top layer map indices
        :param code_b: bottom layer map indices
        :return: decoded tensor in input space
        """
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
