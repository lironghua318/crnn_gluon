
"""
LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE (1998)
"""
import mxnet as mx
from ..config import config
from mxnet.gluon import nn
from mxnet.gluon import rnn
# from .simplenet import SimpleNet


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
        with self.name_scope():
            self.conv = nn.HybridSequential(prefix='')
            with self.conv.name_scope():
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

class BidirectionalLSTM(nn.HybridBlock):
    def __init__(self):
        super(BidirectionalLSTM, self).__init__()
        with self.name_scope():
            self.net_lstm = nn.HybridSequential(prefix='')
            with self.net_lstm.name_scope():
                self.net_lstm.add(rnn.LSTM(hidden_size=config.num_hidden,num_layers=config.num_lstm_layer, \
                                   layout='TNC',bidirectional=True))
            # self.fc = nn.Dense(units=nOut, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.net_lstm(x)
        # x = self.fc(x)  # [T * b, nOut]
        return x



class CRNN(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(CRNN, self).__init__(**kwargs)
        # self.network = network
        with self.name_scope():
            self.net = SimpleNet()
            self.net_lstm = BidirectionalLSTM()
            self.fc = nn.Dense(units=config.num_classes, flatten=True)

    def hybrid_forward(self, F, x):
        net = self.net(x)#net,b,c,h,w
        net = F.transpose(data=net, axes=[3, 2, 0, 1])  # whbc
        net = F.reshape(data=net, shape=(0,-3, 0))#w(hb)c
        net = self.net_lstm(net)
        net = F.reshape(data=net,shape = (-3,0))#(whb)c
        pred = self.fc(net)  # (bz x 25) x num_classes
        return pred

if __name__ == '__main__':
    print(mx.__version__)
    ctx = mx.cpu()
    a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    net = CRNN()
    # net = VGG()
    # net.hybridize()
    net.initialize(ctx=ctx)
    b = net(a)
    print(b.shape)
    print(net)
    # print(net.summary(a))

