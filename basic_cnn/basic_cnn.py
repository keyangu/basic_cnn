# coding: utf-8

from collections import OrderedDict
import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class CNN(Chain):
    def __init__(self, out_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=64, ksize=(4, 4))
            self.bnorm1 = L.BatchNormalization(size=64)
            self.conv2 = L.Convolution2D(in_channels=None, out_channels=128, ksize=(4, 4))
            self.bnorm2 = L.BatchNormalization(size=128)
            self.conv3 = L.Convolution2D(in_channels=None, out_channels=256, ksize=(4, 4))
            self.bnorm3 = L.BatchNormalization(size=256)
            self.l1 = L.Linear(in_size=None, out_size=1024)
            self.l2 = L.Linear(in_size=None, out_size=512)
            self.l3 = L.Linear(in_size=None, out_size=out_num)
        
    def layers(self):
        return OrderedDict(
            conv1 = [self.conv1, self.bnorm1, F.relu, _max_pooling_2d],
            conv2 = [self.conv2, self.bnorm2, F.relu, _max_pooling_2d],
            conv3 = [self.conv3, self.bnorm3, F.relu, _max_pooling_2d],
            lnr1 = [self.l1, F.dropout, F.relu],
            lnr2 = [self.l2, F.relu],
            lnr3 = [self.l3],
            prob = [F.softmax]
        )
        
    def extract(self, x, layers=None):
        if layers == None:
            layers = ['lnr3']
        
        h = x
        ret = {}
        for k, v in self.layers().items():
            for func in v:
                h = func(h)
            if k in layers:
                ret[k] = h
        return ret
        
    def forward(self, x):
        return self.extract(x)['lnr3']

    def predict(self, x):
        return self.extract(x, ['prob'])['prob']

def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)