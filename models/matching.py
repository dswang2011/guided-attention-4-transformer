# -*- coding: utf-8 -*-
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras import backend as K
from keras.layers import Layer,Dropout
import numpy as np
import tensorflow as tf
import keras
def identity_loss(y_true, y_pred):

    return K.mean(y_pred)


def precision_batch(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_pred,0),"float32"))

def getOptimizer(name="sgd",lr=0.0001):
    name=name.strip().lower()
    if name=="sgd":
        optimizer=keras.optimizers.SGD(lr=lr*0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif name=="rmsprop":
        optimizer=keras.optimizers.RMSprop(lr=lr*0.001, rho=0.9, epsilon=None, decay=0.0)
    elif name=="adagrad":
        optimizer=keras.optimizers.Adagrad(lr=lr*0.01, epsilon=None, decay=0.0)
    elif name=="adadelta":
        optimizer=keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
    elif name=="adam":
        optimizer=keras.optimizers.Adam(lr=lr*0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)     
    elif name=="adamax":
        optimizer=keras.keras.optimizers.Adamax(lr=lr*0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)     
    elif name=="nadam":
        optimizer=keras.optimizers.Nadam(lr=lr*0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    else:
        raise Exception("optimizer not supported: {}, only support sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam".format(name))
    return optimizer


class MarginLoss(Layer):

    def __init__(self, margin = 1, **kwargs):
        # self.output_dim = output_dim
        self.margin = margin
        super(MarginLoss, self).__init__(**kwargs)

    def get_config(self):
        config = {'margin': self.margin}
        base_config = super(MarginLoss, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(MarginLoss, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        score1,score2 = inputs

        output = K.maximum(score2-score1+self.margin,0)
        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        
        return input_shape[0]





class Stack(Layer):

    def __init__(self, dropout_keep_prob = 1, axis = -1, keep_dims = True, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        self.dropout_keep_prob=dropout_keep_prob
        self.dropout_probs = Dropout(dropout_keep_prob)
        super(Stack, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(Stack, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

    
        super(Stack, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        x = inputs


        output= K.transpose(K.stack([x,1-x]))
        return output

    def compute_output_shape(self, input_shape):
#        print(input_shape)
        # print(type(input_shape[1]))
        output_shape = [input_shape[0],2]
        
#        print(output_shape)
        return([tuple(output_shape)])




class Cosine(Layer):

    def __init__(self, dropout_keep_prob = 1, axis = -1, keep_dims = True, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        self.dropout_keep_prob=dropout_keep_prob
        self.dropout_probs = Dropout(dropout_keep_prob)
        super(Cosine, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(Cosine, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(Cosine, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        x,y = inputs

        norm1 = K.sqrt(0.00001+ K.sum(x**2, axis = self.axis, keepdims = False))
        norm2 = K.sqrt(0.00001+ K.sum(y**2, axis = self.axis, keepdims = False))
        output= K.sum(self.dropout_probs(x*y),1) / norm1 /norm2


        return output

    def compute_output_shape(self, input_shape):
#        print(input_shape)
        # print(type(input_shape[1]))
        output_shape = []
        if self.axis<0:
            self.axis = len(input_shape[0])+self.axis 
        for i in range(len(input_shape[0])):            
            if not i == self.axis:
                output_shape.append(input_shape[0][i])
        if self.keep_dims:
            output_shape.append(1)
#        print('Input shape of L2Norm layer:{}'.format(input_shape))
#        print(output_shape)
        return([tuple(output_shape)])
        
class Attention(Layer):

    def __init__(self, delta =0.5,c=1,dropout_keep_prob = 1, mean="geometric",axis = -1, keep_dims = True,nb_classes =2, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        self.keep_dims = keep_dims
        self.delta = delta
        self.c = c
        self.mean=mean
        super(Attention, self).__init__(**kwargs)



    def get_config(self):
        config = {'axis': self.axis, 'keep_dims': self.keep_dims}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        x,y = inputs

#        norm1 = K.sqrt(0.00001+ K.sum(x**2, axis = self.axis, keepdims = False))
#        norm2 = K.sqrt(0.00001+ K.sum(y**2, axis = self.axis, keepdims = False))
#        output= K.sum(self.dropout_probs(x*y),1) / norm1 /norm2
        multipled = x*y
        weight = K.softmax(multipled,axis=-1)
        
        representations = K.concatenate([x, y,multipled,multipled*weight], axis=-1)
        return representations

    def compute_output_shape(self, input_shape):
#        print(input_shape)
        # print(type(input_shape[1]))
#        output_shape = []
#        if self.axis<0:
#            self.axis = len(input_shape[0])+self.axis 
#        for i in range(len(input_shape[0])):            
#            if not i == self.axis:
#                output_shape.append(input_shape[0][i])
#        if self.keep_dims:
#            output_shape.append(self.nb_classes) ############
#        print('Input shape of L2Norm layer:{}'.format(input_shape))
#        print(output_shape)
        none_batch , dim = input_shape[0]
        dim = dim * 4
        return([tuple([none_batch,dim])])