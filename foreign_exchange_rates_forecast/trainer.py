import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import AttnEncoder, AttnDecoder,Model
from dataset import Dataset
from torch import optim
import config
import pickle

class Trainer:

    def __init__(self, driving, target, time_step, split, lr, regression=True):
        self.dataset = Dataset(T = time_step, split_ratio=split,binary_file=config.BINARY_DATASET_HEADER)
        self.encoder = AttnEncoder(input_size=self.dataset.get_num_features(), hidden_size=config.ENCODER_HIDDEN_SIZE, time_step=time_step)
        self.decoder = AttnDecoder(code_hidden_size=config.ENCODER_HIDDEN_SIZE, hidden_size=config.DECODER_HIDDEN_SIZE, time_step=time_step)
        self.model = Model(self.encoder,self.decoder)
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.model = self.model.cuda()

        self.model_optim = optim.Adam(self.model.parameters(),lr)
        # self.encoder_optim = optim.Adam(self.encoder.parameters(), lr)
        # self.decoder_optim = optim.Adam(self.decoder.parameters(), lr)

        if(regression):
            # regression model
            self.loss_func = nn.MSELoss()
        else:
            # classification model
            weight = torch.Tensor([1,1])
            # weight = weight.cuda()
            self.loss_func = nn.CrossEntropyLoss(reduce=False, size_average=False, weight = weight)

        self.train_size, self.test_size, self.total_size = self.dataset.get_size()
        print("train_size = %d (in terms of number of binary files)" % self.train_size)
        print("test_size = %d (in terms of number of binary files)" % self.test_size)

    def train_minibatch(self, num_epochs, batch_size, interval, cout, regression=True):
        #x_train, y_train, y_seq_train = self.dataset.get_train_set()
        already_trained = 100
        best_model = -1
        best_correctness = 0
        for epoch in range(num_epochs):
            for file_num in range(self.train_size):
                x_train, y_train, y_seq_train = self.dataset.get_train_set(file_num)
                i = 0
                loss_sum = 0
                while (i < config.MAX_SINGLE_FILE_LINE_NUM):
                    # self.encoder_optim.zero_grad()
                    # self.decoder_optim.zero_grad()
                    self.model_optim.zero_grad()
                    batch_end = i + batch_size
                    if (config.SPLIT_RATIO != 1.0 and file_num == self.train_size - 1 and batch_end > (config.MAX_SINGLE_FILE_LINE_NUM - config.VALIDATION_LINE_NUM)):
                        break
                    if (batch_end > config.MAX_SINGLE_FILE_LINE_NUM):
                        break
                        #batch_end = self.train_size
                    var_x = self.to_variable(x_train[i: batch_end])
                    var_y = Variable(torch.from_numpy(y_train[i: batch_end]).float())
                    var_y_seq = self.to_variable(y_seq_train[i: batch_end])
                    #making sure the driving series has 3 dimensions
                    if var_x.dim() == 2:
                        var_x = var_x.unsqueeze(2)
                    # code = self.encoder(var_x)
                    # y_res = self.decoder(code, var_y_seq)
                    y_res,y_var = self.model(var_x,var_y_seq)
                    # m = torch.distributions.Normal(loc = y_loc,scale=y_var)
                    # loss = torch.sum(-m.log_prob(var_y.unsqueeze(0)))
                    if(regression):
                        # regression model
                        loss = self.loss_func(y_res, var_y)
                    else:
                        # classiication model
                        var_y = var_y.long().cuda()
                        print("y_res.requires_grad: ")
                        print(y_res.requires_grad)
                        print("y_res.type()")
                        print(y_res.type())
                        print("y_res.shape")
                        print(y_res.shape)

                        print("var_y.requires_grad: ")
                        print(var_y.requires_grad)
                        print("var_y.type()")
                        print(var_y.type())
                        print("var_y.shape")
                        print(var_y.shape)
                        loss = self.loss_func(y_res, var_y)

                    loss.backward()
                    # self.encoder_optim.step()
                    # self.decoder_optim.step()
                    self.model_optim.step()
                    if cont:
                        print('epoch[%d], file[%d], batch[%d], loss is %f' % (already_trained + epoch + 1, file_num, batch_end / batch_size, 10000 * loss.data[0]))
                    else:
                        print('epoch[%d], file[%d], batch[%d], loss is %f' % (epoch + 1, file_num, batch_end / batch_size, 10000 * loss.data[0]))
                    loss_sum += loss.data.item()
                    i = batch_end
            if cont:
                print('epoch [%d] finished, the average loss is %f' % (already_trained + epoch + 1, loss_sum))
                if (epoch + 1) % (interval) == 0 or epoch + 1 == (num_epochs + already_trained):
                    torch.save(self.encoder.state_dict(), 'models/30min/encoder_EURUSD_30min_multifile_with_vali' + str(already_trained + epoch + 1) + '.model')
                    torch.save(self.decoder.state_dict(), 'models/30min/decoder_EURUSD_30min_multifile_with_vali' + str(already_trained + epoch + 1) + '.model')
            else:
                print('epoch [%d] finished, the average loss is %f' % (epoch + 1, loss_sum))
                if (epoch + 1) % (interval) == 0 or epoch + 1 == num_epochs:
                    torch.save(self.encoder.state_dict(), 'models/EURUSD/encoder_EURUSD_30min_multifile_with_vali_without_normalization_final_test_new_' + str(epoch + 1) + '.model')
                    torch.save(self.decoder.state_dict(), 'models/EURUSD/decoder_EURUSD_30min_multifile_with_vali_without_normalization_final_test_new_' + str(epoch + 1) + '.model')
            x_vali, y_vali, y_seq_vali = self.dataset.get_validation_set()
            y_pred_validation = self.predict(x_vali, y_vali, y_seq_vali, batch_size)
            seq_len = len(y_vali)
            gt_direction = (y_vali[1:] - y_vali[:seq_len -1])>0
            pred_direction = (y_pred_validation[1:] - y_vali[:seq_len - 1])>0
            correct = np.sum(gt_direction == pred_direction)
            print('number of correct in validation set = %d' % correct)
            print('length of validation set = %d' % seq_len )
            correct = correct/(seq_len - 1)
            if (correct > best_correctness):
                best_model = epoch + 1
                best_correctness = correct
            print('epoch[%d] finished, current correctness is %f, best model so far is model %d with correctness %f' % (epoch + 1, correct, best_model, best_correctness))

    def test(self, num_epochs, batch_size):
        start = self.train_size
        end = self.total_size
        for index in range(start, end, 1):
            #print('testing on part %d' % index)

            #x_train, y_train, y_seq_train = self.dataset.get_train_set(index)
            x_test, y_test, y_seq_test = self.dataset.get_test_set(index)
            # y_pred_train = self.predict(x_train, y_train, y_seq_train, batch_size)
            # f = open('y_train','wb')
            # pickle.dump(y_train,f)
            # f.close()
            # f = open('y_pred_train','wb')
            # pickle.dump(y_pred_train,f)
            # f.close()

            #



            y_pred_test = self.predict(x_test, y_test, y_seq_test, batch_size)
            #print(y_test)
            #print(y_pred_test)
            f = open('y_test_attention_weight_observation_epoch_' + str(num_epochs) + '_part' + str(index - start + 1),'wb')
            pickle.dump(y_test,f)
            f.close()
            f = open('y_pred_test_attention_weight_observation_epoch_' + str(num_epochs) + '_part' + str(index - start + 1),'wb')
            pickle.dump(y_pred_test,f)
            f.close()



            plt.figure()
            # plt.plot(range(1, 1 + self.train_size), y_train, label='train')
            # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_test[:self.test_size//50], label='ground truth')
            plt.plot(range(1 + index * config.MAX_SINGLE_FILE_LINE_NUM, 1 + index * config.MAX_SINGLE_FILE_LINE_NUM + len(y_test)//2), y_test[:len(y_test)//2], label='ground truth')
            # plt.plot(range(1, 1 + self.train_size), y_pred_train, label.='predicted train')
            # plt.plot(range(1, 1 + self.train_size), y_pred_train, label.='predicted train')
            # plt.plot(range(1 + self.train_size, 1 + self.train_size + self.test_size//50), y_pred_test[:self.test_size//50], label='predicted test')
            plt.plot(range(1 + index * config.MAX_SINGLE_FILE_LINE_NUM, 1 + index * config.MAX_SINGLE_FILE_LINE_NUM + len(y_test)//2), y_pred_test[:len(y_test)//2], label='predicted test')
            plt.legend()
            plt.savefig('res-attention_weight_observation_epoch' + str(num_epochs) +'_part_' + str(index - start + 1)+ '.png')
            


    def predict(self, x, y, y_seq, batch_size):
        y_pred = np.zeros(x.shape[0])
        i = 0
        while (i < x.shape[0]):
            #print('testing on batch %d' % (i / batch_size))
            batch_end = i + batch_size
            if batch_end > x.shape[0]:
                break
                #batch_end = x.shape[0]
            var_x_input = self.to_variable(x[i: batch_end])
            var_y_input = self.to_variable(y_seq[i: batch_end])
            if var_x_input.dim() == 2:
                var_x_input = var_x_input.unsqueeze(2)
            # code = self.encoder(var_x_input)
            # y_res = self.decoder(code, var_y_input)
            y_res,_ = self.model(var_x_input,var_y_input)
            for j in range(i, batch_end):
                y_pred[j] = y_res[j - i]
            i = batch_end
        return y_pred

    def load_model(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=lambda storage, loc: storage))
        self.model = Model(self.encoder,self.decoder)
    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())



def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, required=True,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=1,
        help='the mini-batch size')
    parser.add_argument(
        '-s', '--split', type=float, default=0.9,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.0001,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-c', '--cont', action = 'store_true',
        help = 'continue training or new task')
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    cont = args.cont
    trainer = Trainer(config.DRIVING, config.TARGET,config.TIME_STEP, config.SPLIT_RATIO, lr)

    # for index in range(trainer.dataset.get_size()[2]):
    #     f = open(config.BINARY_DATASET_DIR + str(index), 'rb')
    #     temp1 = pickle.load(f)
    #     f = open(config.BINARY_DATASET_DIR + 'vali_' + str(index), 'rb')
    #     temp2 = pickle.load(f)
    #     print('part %d' % index)
    #     print(np.array_equal(temp1[0], temp2[0]))
    #     print(np.array_equal(temp1[1], temp2[1]))
    #     print(np.array_equal(temp1[2], temp2[2]))
    # f = open(config.BINARY_DATASET_DIR + 'vali_validation', 'rb')
    # temp1 = pickle.load(f)
    # f = open(config.BINARY_DATASET_DIR + 'validation', 'rb')
    # temp2 = pickle.load(f)
    # print('validation part')
    # print(np.array_equal(temp1[0], temp2[0]))
    # print(np.array_equal(temp1[1], temp2[1]))
    # print(np.array_equal(temp1[2], temp2[2]))

    if not test:
        if cont:
            trainer.load_model('models/15min/encoder_EURUSD_15min_multifile_100.model', 'models/15min/decoder_EURUSD_15min_multifile_100.model')
            print('continue training for %d epochs' % num_epochs)
        trainer.train_minibatch(num_epochs, batch_size, interval, cont)
    else:
        trainer.load_model('models/encoder_EURUSD_30min_multifile_with_vali_144.model', 'models/decoder_EURUSD_30min_multifile_with_vali_144.model')
        trainer.test(num_epochs, batch_size)
