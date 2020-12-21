import numpy as np
import keras
import collections
import pickle
import os


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, opt, split, list_IDs, labels, n_channels=1, shuffle=False):
        self.opt = opt
        self.split = split
        self.root = 'datasets/' 
        # 'Initialization'
        self.dim = self.opt.embedding_dim
        self.batch_size = self.opt.batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs = list_IDs[:len(self.list_IDs)//2]    # half the data
        self.n_channels = n_channels
        self.n_classes = self.opt.nb_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        # my data
        self.data = {}

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)

        # 
        min_id = int(list_IDs_temp[0].split('-')[1])
        max_id = int(list_IDs_temp[-1].split('-')[1])
        # 0-64;    19998-110023 -> min_id = 19,998, max_id = 110023;  file_index = 10, next_index = 11
        file_index, start_index = min_id//10000, min_id%10000
        next_index, end_index = max_id//10000, max_id%10000
            
        if file_index not in self.data or len(self.data[file_index])==0: 
            self.data[file_index] = pickle.load(open(os.path.join(self.root,self.opt.dataset,self.split+'_'+str(file_index*10000)+'.pkl'),'rb'))
        if next_index not in self.data or len(self.data[next_index])==0: 
            self.data[next_index] = pickle.load(open(os.path.join(self.root,self.opt.dataset,self.split+'_'+str(next_index*10000)+'.pkl'),'rb'))

        # Generate data
        if file_index==next_index:
            x1,x_tag1, masks1 = self.data[file_index]
            x_seq = x1[start_index:end_index+1]
            m0 = masks1[0][start_index:end_index+1]
            m1 = masks1[1][start_index:end_index+1]
            m2 = masks1[2][start_index:end_index+1]
            m3 = masks1[3][start_index:end_index+1]
            m4 = masks1[4][start_index:end_index+1]
            
            x = [x_seq,m0,m1,m2,m3,m4]
            if self.opt.tag_encoding==1:
                tag_seq = x_tag1[start_index:end_index+1]
                x = [x_seq,tag_seq]

            # remove history
            if end_index+1==10000: 
                del self.data[file_index]

        else:
            x1,x_tag1, masks1 = self.data[file_index]
            x2,x_tag2, masks2 = self.data[next_index]
            # prepare
            x_seq = np.concatenate((x1[start_index:], x2[:end_index+1]), axis=0)
            m0 = np.concatenate((masks1[0][start_index:],masks2[0][:end_index+1]), axis=0)
            m1 = np.concatenate((masks1[1][start_index:], masks2[1][:end_index+1]), axis=0)
            m2 = np.concatenate((masks1[2][start_index:], masks2[2][:end_index+1]), axis=0)
            m3 = np.concatenate((masks1[3][start_index:], masks2[3][:end_index+1]), axis=0)
            m4 = np.concatenate((masks1[4][start_index:], masks2[4][:end_index+1]), axis=0)

            x = [x_seq,m0,m1,m2,m3,m4]

            if self.opt.tag_encoding==1:
                tag_seq = np.concatenate((x_tag1[start_index:], x_tag2[:end_index+1]), axis=0)
                x = [x_seq,tag_seq]
                
            # remove history
            del self.data[file_index]       

        y = []
        for i, ID in enumerate(list_IDs_temp):
            y.append(self.labels[ID])
        return x, np.asarray(y)