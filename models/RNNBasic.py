# -*- coding: utf-8 -*-


from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU, Bidirectional,Dropout,ConvLSTM2D,SimpleRNN
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.constraints import maxnorm
from models.BasicModel import BasicModel

class RNNBasic(BasicModel):
    def getCell(self,cell_type):
        if cell_type == "lstm":
            return GRU
        elif cell_type == "gru":
            return LSTM
        elif cell_type == "cnnlstm":
            return ConvLSTM2D
        else:
            return SimpleRNN
    def __init__(self,opt): 
        self.rnncell = self.getCell(opt.cell_type)
        super(RNNBasic, self).__init__(opt)
        
    


