import mxnet as mx
from mxnet.gluon import nn
from ..config import config

class SimpleNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(SimpleNet, self).__init__(**kwargs)
        kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        padding_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        layer_size = [min(32 * 2 ** (i + 1), 512) for i in range(len(kernel_size))]

        def convRelu(layer, i, bn=True):
            layer.add(nn.Conv2D(channels=layer_size[i], kernel_size=kernel_size[i],padding=padding_size[i]))
            if bn:
                layer.add(nn.BatchNorm())
            layer.add(nn.LeakyReLU(alpha=0.25))
            return layer
        self.conv = nn.HybridSequential(prefix='')
        self.conv = convRelu(self.conv,0)  # bz x 64 x 32 x 280

        self.max_pool = nn.HybridSequential(prefix='')
        self.max_pool.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # bz x 128 x 16 x 140
        self.avg_pool = nn.HybridSequential(prefix='')
        self.avg_pool.add(nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2))) # bz x 128 x 16 x 140

        self.net = nn.HybridSequential(prefix='')
        self.net = convRelu(self.net,1)
        self.net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # bz x 256 x 8 x 70
        self.net = convRelu(self.net,2, True)
        self.net = convRelu(self.net,3)
        self.net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # bz x 512 x 4 x 35
        self.net = convRelu(self.net,4, True)
        self.net = convRelu(self.net,5)
        self.c = 512
        if not config.no4x1pooling:
            self.cols_pool = nn.HybridSequential(prefix='')
            self.cols_pool.add(nn.AvgPool2D(pool_size=(4, 1)))  # bz x 512 x 1 x 35
            self.cols_pool.add(nn.Dropout(rate=0.5))
        else:
            self.c =self.c * 4
            self.no_cols_pool = nn.HybridSequential(prefix='')
            self.no_cols_pool.add(nn.Dropout(rate=0.5))

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        max = self.max_pool(x)
        avg = self.avg_pool(x)
        x = max - avg
        x = self.net(x)
        if not config.no4x1pooling:
            x = self.cols_pool(x)
        else:
            x = self.no_cols_pool(x)
            x = F.reshape(data=x, shape=(0, -3, 1, -2))
        return x





