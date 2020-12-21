
# -*- coding: utf-8 -*-

from models.CNN import CNN
from models.BiLSTM import BiLSTM
from models.Transformer import Transformer
from models.GAH import GAH
from models.GAHs import GAHs


def setup(opt):
    if opt.model == "cnn":
        model = CNN(opt)
    elif opt.model == "bilstm":
        model = BiLSTM(opt)
    elif opt.model == "transformer":
        model = Transformer(opt)
    elif opt.model == "gah":
        model = GAH(opt)
    elif opt.model =='gahs':
        model = GAHs(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
