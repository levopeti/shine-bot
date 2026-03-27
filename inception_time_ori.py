import torch
from torch.nn import Module, Conv1d, MaxPool1d, BatchNorm1d, ReLU, Sequential, AdaptiveAvgPool1d, Linear


class InceptionModule(Module):
    def __init__(self,
                 in_channels: int,
                 n_filters: int,
                 kernel_sizes: tuple = (9, 19, 39),
                 bottleneck_channels: int = 32,
                 activation: Module = ReLU()):
        """
        param in_channels:				Number of input channels (input features)
        param n_filters:				Number of filters per convolution layer => out_channels = 4*n_filters
        param kernel_sizes:			List of kernel sizes for each convolution.
                                      Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                      This is necessary because of padding size.
                                      For correction of kernel_sizes use function "correct_sizes".
        param bottleneck_channels:		Number of output channels in bottleneck.
                                      Bottleneck won't be used if number of in_channels is equal to 1.
        param activation:				Activation function for output tensor (nn.ReLU()).
        """
        super().__init__()
        if in_channels > 1:
            # input shape: [n, c, l]
            self.bottleneck = Conv1d(in_channels=in_channels,
                                     out_channels=bottleneck_channels,
                                     kernel_size=1,
                                     stride=1,
                                     bias=False)
        else:
            self.bottleneck = lambda x: x
            bottleneck_channels = 1

        self.max_pool = MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_bottleneck_1 = Conv1d(in_channels=bottleneck_channels,
                                             out_channels=n_filters,
                                             kernel_size=kernel_sizes[0],
                                             stride=1,
                                             padding=kernel_sizes[0] // 2,
                                             bias=False)
        self.conv_from_bottleneck_2 = Conv1d(in_channels=bottleneck_channels,
                                             out_channels=n_filters,
                                             kernel_size=kernel_sizes[1],
                                             stride=1,
                                             padding=kernel_sizes[1] // 2,
                                             bias=False)
        self.conv_from_bottleneck_3 = Conv1d(in_channels=bottleneck_channels,
                                             out_channels=n_filters,
                                             kernel_size=kernel_sizes[2],
                                             stride=1,
                                             padding=kernel_sizes[2] // 2,
                                             bias=False)
        self.conv_from_maxpool = Conv1d(in_channels=in_channels,
                                        out_channels=n_filters,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False)
        self.batch_norm = BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], dim=1)
        Z = self.batch_norm(Z)
        Z = self.activation(Z)
        return Z


class InceptionTime(Module):
    def __init__(self,
                 in_channels: int,
                 out_size: int,
                 n_filters: int = 32,
                 kernel_sizes: tuple = (9, 19, 39),
                 bottleneck_channels: int = 32,
                 use_residual: bool = True,
                 activation: Module = ReLU()):
        super().__init__()
        self.use_residual = use_residual
        self.activation = activation
        self.inception_1 = InceptionModule(in_channels=in_channels,
                                           n_filters=n_filters,
                                           kernel_sizes=kernel_sizes,
                                           bottleneck_channels=bottleneck_channels,
                                           activation=activation)
        self.inception_2 = InceptionModule(in_channels=4 * n_filters,
                                           n_filters=n_filters,
                                           kernel_sizes=kernel_sizes,
                                           bottleneck_channels=bottleneck_channels,
                                           activation=activation)
        self.inception_3 = InceptionModule(in_channels=4 * n_filters,
                                           n_filters=n_filters,
                                           kernel_sizes=kernel_sizes,
                                           bottleneck_channels=bottleneck_channels,
                                           activation=activation)
        if self.use_residual:
            self.residual = Sequential(Conv1d(in_channels=in_channels,
                                              out_channels=4 * n_filters,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                       BatchNorm1d(num_features=4 * n_filters))
        self.global_avg_pool = AdaptiveAvgPool1d(1)
        self.linear = Linear(4 * n_filters, out_size)

    def forward(self, x):
        z = self.inception_1(x)
        z = self.inception_2(z)
        z = self.inception_3(z)
        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)
        z = self.global_avg_pool(z)
        z = torch.squeeze(z)
        z = self.linear(z)
        return z