import numpy as np
import pandas as pd
import math
import config
import pickle
import os.path
class Dataset:
    def __init__(self, driving_csv=[], target_csv='', T=20, split_ratio=0.9,binary_file = ''):
        self.split_ratio = split_ratio

        if(os.path.isfile(binary_file)):
            print("binary file already exists, loading...\n")
            self.load(binary_file)
        else:
            end = 0
            print("creating binary file...\n")
            FXrate_frame_x = []
            self.gran = config.GRANULARITY
            for i in range(len(driving_csv)):
                col = pd.read_csv(driving_csv[i])['Close'].fillna(method='pad')
                if(i == 0):
                    end = (len(col) - config.TIME_STEP * max(self.gran)) / min(self.gran)
                    end = end / config.MAX_SINGLE_FILE_LINE_NUM + 1
                    end = int(end * config.SPLIT_RATIO)
                    end = (end * config.MAX_SINGLE_FILE_LINE_NUM - config.VALIDATION_LINE_NUM) * min(self.gran) + config.TIME_STEP * max(self.gran)
                    if(config.SPLIT_RATIO == 1.0):
                        end = config.MAX_SINGLE_FILE_LINE_NUM * 10 * min(self.gran) + max(self.gran) * config.TIME_STEP
                    print("end = %d" % (end))
                    print("len(col) = %d" % (len(col)))
                col = (col - col[:end].min())/(col[:end].max() - col[:end].min())
                FXrate_frame_x.append(col.values)
            FXrate_frame_x = np.array(FXrate_frame_x)
            y = pd.read_csv(target_csv)['Close'].fillna(method = 'pad')
            FXrate_frame_y = (y - y[:end].min())/(y[:end].max() - y[:end].min())
            self.X = []
            self.y = []
            self.y_seq = []
            self.is_header = True
            self.time_series_gen(FXrate_frame_x,FXrate_frame_y,T,True)

    def get_size(self):
        return self.train_size, self.test_size, self.total_size
    def get_train_set(self, index):
        #train_size,_ = self.get_size()
        f = open(config.BINARY_DATASET_DIR + str(index),'rb')
        ds = pickle.load(f)
        f.close()
        return ds[0], ds[1], ds[2]
    def get_test_set(self, index):
        #train_size,_ = self.get_size()
        f = open(config.BINARY_DATASET_DIR + str(index),'rb')
        ds = pickle.load(f)
        f.close()
        return ds[0], ds[1], ds[2]
        #return self.X[train_size:], self.y[train_size:], self.y_seq[train_size:]
    def get_num_features(self):
        return self.num_features
    def get_validation_set(self):
        f = open(config.BINARY_DATASET_DIR + 'validation', 'rb')
        ds = pickle.load(f)
        f.close()
        return ds[0], ds[1], ds[2]
    # This function wont be called if loading training data from binary file
    def time_series_gen(self,X,y,T,shuffle = False, regression=True):
        ts_x, ts_y, ts_y_seq = [], [], []
        average = False
        #print(X.shape)
        for i in range(T*max(self.gran),X.shape[1],min(self.gran)):
            #print(i)
            # last = i + T
            col = []
            for g in self.gran:
                #print(g)
                col.append(X[:,i:i-T*g:-g])
                if(average):
                    col2 = []
                    if (g > 1):
                        for index in range (T):
                            end = i - index * g
                            start = i - (index + 1) * g
                            col2.append(np.sum(X[:,start:end], axis = 1)/g)
                        col2 = np.transpose(col2)
                        col.append(col2)
            col = np.array(col)
            #print(col.shape)
            col = col.reshape(-1,col.shape[0]*col.shape[1])
            #print(col.shape)
            ts_x.append(col)
            if(regression):
                # regression model
                ts_y.append(y[i])
            else:
                # classification model
                up = [1,0]
                down = [0,1]
                if((y[i] - y[i - min(self.gran)]) > 0):
                    ts_y.append(1)
                else:
                    ts_y.append(0)

            ts_y_seq.append(y[i-T:i])
            #print(ts_y)
        print("length of ts_x is %d" % len(ts_x))
        if(i == 1):
            print(ts_x[0].shape)
        print("length of ts_y is %d" % len(ts_y))
        print("length of ts_seq is %d" % len(ts_y_seq))
        self.total_size = int(len(ts_x) / config.MAX_SINGLE_FILE_LINE_NUM) + 1
        self.train_size = int(self.split_ratio * self.total_size)
        if(config.SPLIT_RATIO == 1.0):
            self.train_size = self.train_size - 1
        self.test_size = self.total_size - self.train_size
        self.num_features = np.array(ts_x[0:2]).shape[2]
        print("total_size is %d" % self.total_size)
        print("train_size is %d" % self.train_size)
        print("test_size is %d" % self.test_size)
        print("num_features is %d" % self.num_features)

        if(config.SPLIT_RATIO == 1.0):
            randomize = np.arange(self.train_size * config.MAX_SINGLE_FILE_LINE_NUM)
        else:
            randomize = np.arange(self.train_size * config.MAX_SINGLE_FILE_LINE_NUM - config.VALIDATION_LINE_NUM)
        
        if shuffle:
            print("shuffling training data... \n")
            np.random.shuffle(randomize)
            #print(randomize)
            for i in range(self.train_size):
                start = i * config.MAX_SINGLE_FILE_LINE_NUM
                end = start + config.MAX_SINGLE_FILE_LINE_NUM
                if (end > len(randomize)) :
                    print("end > len(randomize)")
                    end = len(randomize)
                data = [np.array(ts_x)[randomize[start:end]], np.array(ts_y)[randomize[start:end]], np.array(ts_y_seq)[randomize[start:end]]]
                f = open(config.BINARY_DATASET_DIR  + str(i), 'wb')
                pickle.dump(data, f)
                f.close()
                print("dataset part %d, start %d: , end %d" % (i, start, end))
        else:
            print("not shuffling training data... \n")
            randomize = np.arange(config.MAX_SINGLE_FILE_LINE_NUM)
            for i in range(self.train_size):
                start = i * config.MAX_SINGLE_FILE_LINE_NUM
                end = start + config.MAX_SINGLE_FILE_LINE_NUM
                #print(randomize)
                if (end > len(randomize)) :
                    end = len(randomize)
                data = [np.array(ts_x[start:end])[randomize], np.array(ts_y[start:end])[randomize], np.array(ts_y_seq[start:end])[randomize]]
                f = open(config.BINARY_DATASET_DIR  + str(i), 'wb')
                pickle.dump(data, f)
                f.close()
                print("dataset part %d, start %d: , end %d" % (i, start, end))

        print("recording validation data... \n")
        if(config.SPLIT_RATIO == 1.0):
            data = [np.array(ts_x[len(randomize):]), np.array(ts_y[len(randomize):]), np.array(ts_y_seq[len(randomize):])]
        else:
            data = [np.array(ts_x[len(randomize):self.train_size*config.MAX_SINGLE_FILE_LINE_NUM]), np.array(ts_y[len(randomize):self.train_size*config.MAX_SINGLE_FILE_LINE_NUM]), np.array(ts_y_seq[len(randomize):self.train_size*config.MAX_SINGLE_FILE_LINE_NUM])]
        print("validation length: %d" % (len(data[0])))
        f = open(config.BINARY_DATASET_DIR + 'validation', 'wb')
        pickle.dump(data, f)
        f.close()
        print("recording test data... \n")
        for j in range(self.train_size, self.total_size, 1):
            randomize = np.arange(config.MAX_SINGLE_FILE_LINE_NUM)
            start = j * config.MAX_SINGLE_FILE_LINE_NUM
            end = start + config.MAX_SINGLE_FILE_LINE_NUM
            if (end > len(ts_x)):
                end = len(ts_x)
                randomize = np.arange(len(ts_x) - start)
            #print(randomize)
            data = [np.array(ts_x[start:end])[randomize], np.array(ts_y[start:end])[randomize], np.array(ts_y_seq[start:end])[randomize]]
            f = open(config.BINARY_DATASET_DIR + str(j), 'wb')
            pickle.dump(data, f)
            f.close()
            print("dataset part %d, start %d: , end %d" % (j, start, end))
    def save(self,binary_file):
        f = open(binary_file,'wb')
        pickle.dump(self,f)
        f.close()
    def load(self,binary_file):
        f = open(binary_file,'rb')
        ds = pickle.load(f)
        self.X, self.y, self.y_seq, self.total_size, self.train_size, self.test_size, self.num_features = ds.X, ds.y, ds.y_seq, ds.total_size, ds.train_size, ds.test_size, ds.num_features

        f.close()
if __name__ == '__main__':
    dir = config.DATA_DIR
    driving_csv = [
                   dir + 'AUDUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'EURUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'GBPUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'USDCAD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'USDCHF_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'USDJPY_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'XAGUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   dir + 'XAUUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',
                   ]
    ds = Dataset(driving_csv,dir + 'EURUSD_1 Min_Ask_2004.01.01_2017.12.08.csv',T = config.TIME_STEP, split_ratio = config.SPLIT_RATIO)
    # ds = Dataset(binary_file=config.BINARY_DATASET)
    ds.save(config.BINARY_DATASET_HEADER)
