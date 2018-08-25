import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
import pandas as pd
from dataset import Dataset


y_test_filename = 'with_vali_result/y_test_EURUSD_30min_multifile_with_vali_without_normalization_trial3_epoch_111_part'
y_pred_test_filename = 'with_vali_result/y_pred_test_EURUSD_30min_multifile_with_vali_without_normalization_trial3_epoch_111_part'


if __name__ == '__main__':
    file_begin = 1
    file_end = 2
    counter = 0
    y_test = []
    y_test = np.array(y_test)
    y_pred_test = []
    y_pred_test = np.array(y_pred_test)
    gain = 0
    gain_list = []
    snow_ball_gain = 1
    snow_ball_gain_list = []
    train_size = 9

    is_dollar = True
    starting_value = 1.0

    num_of_test = 30
    test_window = 400


    target_csv_ask = config.DATA_DIR + 'EURUSD_1 Min_Ask_2004.01.01_2017.12.08.csv'
    y_ask = pd.read_csv(target_csv_ask)['Close'].fillna(method = 'pad')
    target_csv_bid = config.DATA_DIR + 'EURUSD_1 Min_Bid_2004.01.01_2017.12.08.csv'
    y_bid = pd.read_csv(target_csv_bid)['Close'].fillna(method = 'pad')

    ts_y_ask = []
    ts_y_bid = []
    for i in range(config.TIME_STEP * max(config.GRANULARITY), len(y_ask), min(config.GRANULARITY)):
        ts_y_ask.append(y_ask[i])
        ts_y_bid.append(y_bid[i])

    raw_y_ask = ts_y_ask[train_size * config.MAX_SINGLE_FILE_LINE_NUM:]
    raw_y_bid = ts_y_bid[train_size * config.MAX_SINGLE_FILE_LINE_NUM:]
    print('len(raw_y_ask) = %d' % len(raw_y_ask))

    for index in range(file_begin, file_end + 1, 1):
        gt = open(y_test_filename + str(index), 'rb')
        pred = open(y_pred_test_filename + str(index), 'rb')
        temp = pickle.load(gt)
        temp = np.array(temp)
        y_test = np.concatenate((y_test, temp))

        temp = pickle.load(pred)
        temp = np.array(temp)
        y_pred_test = np.concatenate((y_pred_test, temp))
        gt.close()
        pred.close()
 

    seq_len = len(y_test)
    print('len(y_test) = %d' % seq_len)

    test_start = np.arange(seq_len - test_window)
    np.random.shuffle(test_start)
    test_start = test_start[:num_of_test]
    print(test_start)

    gt_direction = (y_test[1:] - y_test[:seq_len -1])>0
    pred_direction = (y_pred_test[1:] - y_test[:seq_len - 1])>0
    correct = np.sum(gt_direction == pred_direction)
    print("difference is %f " % np.sum(y_test - y_pred_test))
    #print(correct)
    correct = correct/(seq_len - 1)
    print("test set direction correct percentage: " + str(correct))

    for i in range(1,len(raw_y_ask)):
        if(y_pred_test[i] != 0):
            # other currency appreciate
            if ((y_pred_test[i] - y_test[i - 1]) > 0):
                # least optimal
                gain = gain + (1 / raw_y_ask[i - 1] * raw_y_bid[i] - 1)
                # most optimal
                # gain = gain + (1 / raw_y_bid[i - 1] * raw_y_ask[i] - 1)
                snow_ball_gain = snow_ball_gain * raw_y_bid[i] / raw_y_ask[i - 1]
                counter += 1
                if(is_dollar):
                    # least optimal
                    starting_value = starting_value / raw_y_ask[i - 1]
                    # most optimal
                    # starting_value = starting_value / raw_y_bid[i - 1]
                    is_dollar = False
            # other currency depreciate
            elif((y_pred_test[i] - y_test[i - 1]) < 0):
                #gain = gain - (raw_y_ask[i] - raw_y_bid[i - 1])
                #snow_ball_gain = snow_ball_gain * y_test[i - 1] / y_test[i]
                if(not is_dollar):
                    # least optimal
                    starting_value = starting_value * raw_y_bid[i - 1]
                    # most optimal
                    # starting_value = starting_value * raw_y_ask[i - 1]
                    is_dollar = True
                counter += 1
        gain_list.append(gain)
        snow_ball_gain_list.append(snow_ball_gain)

    print("test set total gain: %f" % gain)
    print("test set total snowball gain: %f" % snow_ball_gain)
    #print(gain_list)

    if(is_dollar):
        print("dollar: %f" % starting_value)
    else:
        print("euro: %f" % starting_value)
    plt.figure(figsize=(16,8),dpi=64)
    gain_list = np.array(gain_list)
    plt.plot(range(1, 1 + len(gain_list)//1), gain_list[:len(gain_list)//1], label='gain')
    plt.legend()
    plt.savefig('gain_growth.png')

    plt.figure(figsize=(16,8),dpi=64)
    snow_ball_gain_list = np.array(snow_ball_gain_list)
    plt.plot(range(1, 1 + len(snow_ball_gain_list)//1), snow_ball_gain_list[:len(snow_ball_gain_list)//1], label='snow_ball_gain')
    plt.legend()
    plt.savefig('snow_ball_gain_growth.png')

    for index, start in enumerate(test_start):
        gain = 0
        snow_ball_gain = 1
        is_dollar = True
        starting_value = 1.0
        counter = 0  
        for i in range(start, start + test_window):
            if(y_pred_test[i] != 0):
                # other currency appreciate
                if ((y_pred_test[i] - y_test[i - 1]) > 0):
                    counter = counter + 1
                    # least optimal
                    gain = gain + (1 / raw_y_ask[i - 1] * raw_y_bid[i] - 1)
                    # most optimal
                    # gain = gain + (1 / raw_y_bid[i - 1] * raw_y_ask[i] - 1)
                    snow_ball_gain = snow_ball_gain * raw_y_bid[i] / raw_y_ask[i - 1]
                    if(is_dollar):
                        # least optimal
                        starting_value = starting_value / raw_y_ask[i - 1]
                        # most optimal
                        # starting_value = starting_value / raw_y_bid[i - 1]
                        is_dollar = False
                # other currency depreciate
                elif((y_pred_test[i] - y_test[i - 1]) < 0):
                    counter = counter + 1
                    #gain = gain - (raw_y_ask[i] - raw_y_bid[i - 1])
                    #snow_ball_gain = snow_ball_gain * y_test[i - 1] / y_test[i]
                    if(not is_dollar):
                        # least optimal
                        starting_value = starting_value * raw_y_bid[i - 1]
                        # most optimal
                        # starting_value = starting_value * raw_y_ask[i - 1]
                        is_dollar = True
        if(is_dollar):
            print("test: %d, num of transaction: %d, gain: %f, snow_ball_gain: %f, dollar: %f" % (index + 1, counter, gain, snow_ball_gain, starting_value))
        else:
            print("test: %d, num of transaction: %d, gain: %f, snow_ball_gain: %f, euro: %f, equivalent dollar: %f" % (index + 1, counter, gain, snow_ball_gain, starting_value, starting_value * raw_y_bid[i - 1]))
    #print("counter = %d" % counter)

    #print(snow_ball_gain)
    #print(snow_ball_gain_list)
    #print(len(snow_ball_gain_list))
    #print(len(gain_list))
    #print(y_test)
    #print(y_pred_test[len(y_pred_test) - 20:])
    


