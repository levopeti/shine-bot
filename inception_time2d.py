import torch
from gymnasium import spaces
from torch.nn import Module, Conv1d, Conv2d, MaxPool2d, BatchNorm1d, BatchNorm2d, ReLU, Sequential, AdaptiveAvgPool1d, Linear
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class InceptionModule2D(Module):
    def __init__(self,
                 in_channels: int,
                 n_filters: int,
                 kernel_sizes: tuple = (9, 19, 39),
                 bottleneck_channels: int = 32,
                 activation: Module = ReLU()):
        """
        Input shape: [batch, in_channels, time]
        Belül: [batch, 1, in_channels, time] alakra reshape-eli,
        majd 2D conv-okat alkalmaz (kernel: [1, k] vagy [in_channels, k]),
        hogy minden feature-nek külön filtere legyen.

        Output shape: [batch, 4*n_filters, time]
        """
        super().__init__()
        self.in_channels = in_channels

        # Bottleneck: [batch, 1, C, T] -> [batch, bottleneck_channels, C, T]
        # kernel (1,1): csatornánkénti keverés időben NINCS, csak filter-szám csökkentés
        if in_channels > 1:
            self.bottleneck = Conv2d(
                in_channels=1,
                out_channels=bottleneck_channels,
                kernel_size=(1, 1),
                bias=False
            )
            conv_in = bottleneck_channels
        else:
            self.bottleneck = lambda x: x
            conv_in = 1

        # Minden conv: kernel (in_channels, k) -> lefedi az összes feature-t egyszerre,
        # de a padding csak az időtengelyen van -> output: [batch, n_filters, 1, T]
        self.conv1 = Conv2d(conv_in, n_filters,
                            kernel_size=(in_channels, kernel_sizes[0]),
                            padding=(0, kernel_sizes[0] // 2), bias=False)
        self.conv2 = Conv2d(conv_in, n_filters,
                            kernel_size=(in_channels, kernel_sizes[1]),
                            padding=(0, kernel_sizes[1] // 2), bias=False)
        self.conv3 = Conv2d(conv_in, n_filters,
                            kernel_size=(in_channels, kernel_sizes[2]),
                            padding=(0, kernel_sizes[2] // 2), bias=False)

        # MaxPool ág: 1D-ből jön, külön kezeljük
        self.max_pool = MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_from_maxpool = Conv2d(1, n_filters,
                                        kernel_size=(in_channels, 1),
                                        padding=(0, 0), bias=False)

        self.batch_norm = BatchNorm2d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, x):
        # x: [B, C, T]
        # B, C, T = x.shape

        # 2D-sé alakítás: [B, 1, C, T]
        x_2d = x.unsqueeze(1)

        # Bottleneck: [B, bottleneck_ch, C, T]
        x_bn = self.bottleneck(x_2d)

        # 3 konvolúciós ág: kernel (C, k) -> [B, n_filters, 1, T]
        out1 = self.conv1(x_bn)
        out2 = self.conv2(x_bn)
        out3 = self.conv3(x_bn)

        # MaxPool ág az eredeti x_2d-n
        mp = self.max_pool(x_2d)  # [B, 1, C, T]
        out4 = self.conv_from_maxpool(mp)  # [B, n_filters, 1, T]

        # Összefűzés a filter-dimenzión: [B, 4*n_filters, 1, T]
        out = torch.cat([out1, out2, out3, out4], dim=1)

        out = self.batch_norm(out)
        out = self.activation(out)

        # Visszaalakítás 1D-be: [B, 4*n_filters, T]
        out = out.squeeze(2)
        return out


class InceptionTime2D(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Box,
                 features_dim: int,
                 in_channels: int,
                 n_filters: int = 32,
                 window_h: int = 12,
                 kernel_sizes: tuple = (9, 19, 39),
                 bottleneck_channels: int = 32,
                 use_residual: bool = True,
                 activation: Module = ReLU()):
        super().__init__(observation_space, features_dim)
        self.n_features = in_channels
        self.window = window_h * 12

        self.use_residual = use_residual
        self.activation = activation

        self.inception_1 = InceptionModule2D(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_2 = InceptionModule2D(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )
        self.inception_3 = InceptionModule2D(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation
        )

        if self.use_residual:
            # A residual marad 1D: az input és output is [B, C, T] alakú,
            # az InceptionModule2D squeeze után visszaad [B, 4*n_filters, T]-t.
            self.residual = Sequential(
                Conv1d(in_channels=in_channels,
                       out_channels=4 * n_filters,
                       kernel_size=1,
                       stride=1,
                       padding=0),
                BatchNorm1d(num_features=4 * n_filters)
            )

        self.global_avg_pool = AdaptiveAvgPool1d(1)
        self.linear = Linear(4 * n_filters, features_dim)

    def _reshape_input(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, length) -> (batch, channels, time)
        batch_size = obs.shape[0]
        x = obs.view(batch_size, self.window, self.n_features)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, observations: torch.Tensor):
        x = self._reshape_input(observations)  # [B, C, T]

        z = self.inception_1(x)  # [B, 4F, T]
        z = self.inception_2(z)  # [B, 4F, T]
        z = self.inception_3(z)  # [B, 4F, T]

        if self.use_residual:
            z = z + self.residual(x)  # [B, 4F, T] + [B, 4F, T]
            z = self.activation(z)

        z = self.global_avg_pool(z)  # [B, 4F, 1]
        z = torch.squeeze(z, 2)  # [B, 4F]
        z = self.linear(z)  # [B, features_dim]
        return z
