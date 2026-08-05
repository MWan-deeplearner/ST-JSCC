import math

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from .common_component import Channel, GDN


class BaseJSCC(nn.Module):
    def __init__(self, model_info, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.epoch = nn.Parameter(torch.zeros(1))
        self.chan_type = model_info['chan_type']
        self.encoder = encoder
        self.channel = Channel(self.chan_type)
        self.decoder = decoder

    def forward(self, x, SNR_info=5):
        # input shape = B X C X H X W
        encoder_output = self.encoder(x)
        decoder_input = self.channel(encoder_output, SNR_info)
        decoder_output = self.decoder(decoder_input)
        return decoder_output

    def get_epoch(self):
        return self.epoch

    def add_epoch(self, number=1):
        with torch.no_grad():
            self.epoch += number
        return self.epoch


class LinearAttentionModular(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)  # 1x1足够
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.eps = 1e-6

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        q = q.reshape(B, C, -1)
        k = k.reshape(B, C, -1)
        v = v.reshape(B, C, -1)
        k_sum = k.sum(dim=-1, keepdim=True)  # [B, C, 1]
        context = torch.matmul(k, v.permute(0, 2, 1))  # [B, C, C]
        out = torch.matmul(context.permute(0, 2, 1), q)  # [B, C, HW]
        out = out / (torch.matmul(k_sum.permute(0, 2, 1), q) + self.eps)
        out = out.reshape(B, C, H, W)
        return self.proj_out(out)


class ResidualModular(nn.Module):
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=1
    ):
        super(ResidualModular, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride = 1, padding=0, bias=False
        )
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        x = self.conv3(x)
        out = out + x
        out = self.prelu(out)
        return out


class GatedFeedForwardNetwork(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.dim = dim
        self.project_in = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class STBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.lam = LinearAttentionModular(in_channels)
        self.rm = ResidualModular(
            in_channels, in_channels, kernel_size=5, stride=1, padding=2
        )
        self.cat_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gffn = GatedFeedForwardNetwork(in_channels)

    def forward(self, x):
        x1 = self.lam(x)
        x2 = self.rm(x)
        x = self.cat_conv(torch.cat([x1, x2], dim=1)) + x
        x = self.gffn(x) + x
        return x


class STEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i >= 5 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 4 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder(model_info),
            STDecoder(model_info)
        )


class STEncoder_32_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 32
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder_32_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 32
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i >= 5 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 4 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC_32_channel(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder_32_channel(model_info),
            STDecoder_32_channel(model_info)
        )


class STEncoder_64_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 64
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder_64_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 64
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i >= 5 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 4 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC_64_channel(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder_64_channel(model_info),
            STDecoder_64_channel(model_info)
        )


class STEncoder_16_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 16
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder_16_channel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 16
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i >= 5 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 4 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC_16_channel(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder_16_channel(model_info),
            STDecoder_16_channel(model_info)
        )


class STEncoder_4_blocks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(4):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder_4_blocks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(4):
            if i >= 4 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 3 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC_4_blocks(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder_4_blocks(model_info),
            STDecoder_4_blocks(model_info)
        )


class STEncoder_6_blocks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_downs = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_downs) * (2 ** num_downs) * (1 / self.rcpp))
        self.downsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(6):
            if i < 2:
                self.downsamples.append(
                    nn.Conv2d(
                        3 if i == 0 else self.hidden_channels, self.hidden_channels,
                        kernel_size=5, stride=2, padding=2
                    )
                )
            else:
                self.downsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.head = nn.Conv2d(self.hidden_channels, self.bottleneck_dim, kernel_size=1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STDecoder_6_blocks(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_channels = 3
        self.rcpp = 12
        self.hidden_channels = 48
        num_ups = 2
        self.bottleneck_dim = int(3 * 2 * (2 ** num_ups) * (2 ** num_ups) * (1 / self.rcpp))
        self.upsamples = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(6):
            if i >= 6 - 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        self.hidden_channels, self.in_channels if i == 5 else self.hidden_channels,
                        kernel_size=5, stride=2, padding=2, output_padding=1
                    )
                )
            else:
                self.upsamples.append(nn.Identity())
            self.layers.append(STBlock(self.hidden_channels))
        self.stem = nn.Conv2d(self.bottleneck_dim, self.hidden_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.upsamples[i](x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class STJSCC_6_blocks(BaseJSCC):
    def __init__(self, model_info):
        super().__init__(
            model_info,
            STEncoder_6_blocks(model_info),
            STDecoder_6_blocks(model_info)
        )
