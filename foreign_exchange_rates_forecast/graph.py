import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch import optim
import config
import pickle
import sys
import csv


def main(args):
    start = int(args[2])
    end = int(args[3]) + 1
    good_interval = 100
    num_epoch = int(args[4])
    for index in range(start, end, 1):

        f = open(args[0] + str(index),'rb')
        y_test = pickle.load(f)
        f.close()
        f = open(args[1] + str(index),'rb')
        y_pred_test = pickle.load(f)
        f.close()

        gt_direction = (y_test[1:] - y_test[:len(y_test) - 1])>0
        pred_direction = (y_pred_test[1:] - y_test[:len(y_test) - 1])>0
        correct = np.sum(gt_direction == pred_direction)/(len(y_test) - 1)


        plt.figure(figsize=(24,8),dpi=64)
        # plt.plot(range(1, 1 + self.train_size), y_train, label='train')
        # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_test[:self.test_size//50], label='ground truth')
        #plt.plot(range(1 + index * config.MAX_SINGLE_FILE_LINE_NUM, 1 + index * config.MAX_SINGLE_FILE_LINE_NUM + len(y_test)//1), y_test[:len(y_test)//1], label='ground truth')
        plt.plot(range(1, 1 + len(y_test)//1), y_test[:len(y_test)//1], label='ground truth')
        # plt.plot(range(1, 1 + self.train_size), y_pred_train, label.='predicted train')
        # plt.plot(range(1, 1 + self.train_size), y_pred_train, label.='predicted train')
        # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_pred_test[:self.test_size//50], label='predicted test')
        #plt.plot(range(1 + index * config.MAX_SINGLE_FILE_LINE_NUM, 1 + index * config.MAX_SINGLE_FILE_LINE_NUM + len(y_test)//1), y_pred_test[:len(y_test)//1], label='predicted test')
        plt.plot(range(1 ,1 + len(y_test)//1), y_pred_test[:len(y_test)//1], label='predicted test')
        plt.legend()
        plt.savefig('graph_longevity_test_epoch'+ str(num_epoch) + '_part_'+ str(index) + '.png')
        print(correct)
        #print(np.sum(gt_direction == pred_direction))
        #print(len(y_test))

if __name__ == '__main__':
    test_file_name = sys.argv[1]
    pred_file_name = sys.argv[2]
    file_start = sys.argv[3]
    file_end = sys.argv[4]
    num_epoch = sys.argv[5]
    args = [test_file_name, pred_file_name, file_start, file_end, num_epoch]

    main(args)