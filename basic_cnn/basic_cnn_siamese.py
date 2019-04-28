# coding: utf-8

from basic_cnn import CNN
import chainer.functions as F
from chainer.backends.cuda import cupy as cp

class CNN_Siamese(CNN):
    def __init__(self, out_num):
        super(CNN_Siamese, self).__init__(out_num)

    def forward(self, x0, t0, x1, t1):
        h0 = super(CNN_Siamese, self).forward(x0)
        h1 = super(CNN_Siamese, self).forward(x1)
        
        t = cp.equal(t0, t1).astype(int)
        
        return F.contrastive(h0, h1, t)