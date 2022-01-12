import torch
import torch.nn as nn

from utils import _facify, compute_same_padding


class Encoder(nn.Module):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        latent_dim,
        fac=1,
        device=None,
    ):
        super().__init__()

        encoder_modules = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(512, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            ),
            nn.GroupNorm(1, _facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=2,
                device=device,
            ),
            Residual3DConvBlock(
                in_channels=_facify(512, fac),
                n_filters=_facify(512, fac),
                kernel_size=3,
                stride=1,
                device=device,
            ),
            nn.Flatten(),
        ]
        self.featurizer = nn.Sequential(*encoder_modules)
        # num_features = width * height * depth
        num_features = (width // 4) * (height // 4) * (depth // 4) * _facify(512, fac)
        # num_features = (width // 8) * (height // 8) * (depth // 8) * _facify(1024, fac)
        self.to_latent_mu = nn.Linear(
            in_features=num_features,
            out_features=latent_dim,
        )
        self.to_latent_logvar = nn.Linear(
            in_features=num_features,
            out_features=latent_dim,
        )

    def forward(self, inputs):
        features = self.featurizer(inputs)
        mu = self.to_latent_mu(features)
        logvar = self.to_latent_logvar(features)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        out_channels,
        resolution,
        fac=1,
        device=None,
    ):
        super().__init__()
        embed_condition_blocks = [
            nn.Conv3d(
                in_channels=condition_n_channels,
                out_channels=_facify(512, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            ),
            nn.GroupNorm(1, num_channels=_facify(512, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(512, fac),
                _facify(512, fac),
                kernel_size=3,
                stride=1,
            ),
            Residual3DConvBlock(
                _facify(512, fac),
                _facify(512, fac),
                kernel_size=3,
                stride=1,
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )

        self.pad = nn.ReplicationPad3d(5)
        embed_noise_blocks = [
            nn.Conv3d(
                z_dim // 8,
                _facify(512, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            ),
            nn.LeakyReLU(),
        ]
        self.embed_noise = nn.Sequential(*tuple(embed_noise_blocks))

        to_image_blocks = [
            nn.Conv3d(
                in_channels=_facify(512, fac) + _facify(512, fac),
                out_channels=_facify(1024, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            ),
            nn.GroupNorm(1, num_channels=_facify(1024, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(1024, fac),
                _facify(1024, fac),
                kernel_size=3,
                stride=1,
            ),
            Residual3DConvBlock(
                _facify(1024, fac),
                _facify(1024, fac),
                kernel_size=3,
                stride=1,
            ),
            nn.Conv3d(
                in_channels=_facify(1024, fac),
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=compute_same_padding(1, 1, 1),
            ),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, c):
        # c is torch.Size([64, 63, 16, 16, 16])
        embedded_c = self.embed_condition(c)

        embedded_z = z.view(embedded_c.size(0), z.size(1) // 8, 2, 2, 2)
        embedded_z = self.embed_noise(self.pad(embedded_z))

        combined_input = torch.cat((embedded_c, embedded_z), dim=1)

        out = self.to_image(combined_input)

        return out


class Residual3DConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_size,
        stride,
        trans=False,
        device=None,
    ):
        super(Residual3DConvBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans = trans

        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    in_channels,
                    out_channels=self.n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
        elif self.trans:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels=self.n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        else:
            self.downsample = nn.Identity()
        self.downsample = self.downsample

        same_padding = compute_same_padding(self.kernel_size, self.stride, dilation=1)
        block_elements = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=same_padding,
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=self.n_filters,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding=same_padding,
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_elements))
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs):
        out = self.block(inputs)
        downsampled = self.downsample(inputs)
        result = 0.5 * (out + downsampled)
        result = self.nonlin(result)
        return result
