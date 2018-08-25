import numpy as np
import pandas as pd
import config
import pickle
import sys
import csv

def baseline(decision_ratio, split_ratio, multifile, time_step, num_time_step):
    target_csv = 'FX/USDCAD_1 Min_Ask_2004.01.01_2017.12.08.csv'
    y = pd.read_csv(target_csv)['Close'].fillna(method = 'pad')
    FXrate_frame_y = (y - y.min())/(y.max() - y.min())
    ts_y = []
    if multifile:
        start = int(split_ratio * (int(len(FXrate_frame_y) / config.MAX_SINGLE_FILE_LINE_NUM) + 1)) * config.MAX_SINGLE_FILE_LINE_NUM
    else:
        for i in range(config.TIME_STEP * max(config.GRANULARITY), FXrate_frame_y.shape[0], min(config.GRANULARITY)):
            ts_y.append(FXrate_frame_y[i])
            start = int(split_ratio * len(ts_y)) - 1
            y_test = ts_y[start:]
            start = start * min(config.GRANULARITY) + config.TIME_STEP * max(config.GRANULARITY)
            end = len(ts_y) * min(config.GRANULARITY) + config.TIME_STEP * max(config.GRANULARITY)
        #temp = FXrate_frame_y[start:end:min(config.GRANULARITY)]

        y_test = np.array(y_test)
        #temp = np.array(temp)
        #print('original')
        #print(y_test)
        #print(temp)
        #print(len(y_test))
        #print(len(temp))
        ground_truth = (y_test[1:] - y_test[:len(y_test) - 1]) > 0
        #print(len(ground_truth))
        #print(ground_truth)
        start = start + min(config.GRANULARITY)
        #end = len(FXrate_frame_y)

        #ground_truth = (FXrate_frame_y[start + 1:] - FXrate_frame_y[:end - 1]) > 0

        predict = []
        for index in range(start, end, min(config.GRANULARITY)):
            vote = FXrate_frame_y[index : index - num_time_step * time_step : -time_step]
            vote = np.array(vote)
            vote = vote.reshape(-1,vote.shape[0])
            vote = vote.squeeze(0)
            vote = vote[::-1]
            #print(index)
            #print(vote)
            rise_fall_sequence = (vote[1:] - vote[:len(vote) - 1]) > 0
            predict = np.append(predict, np.sum(rise_fall_sequence) > (len(vote) * decision_ratio))

        #print(len(predict))
        #print(len(ground_truth))
        #print(ground_truth[:20])
        #print(predict[:20])
        result = np.sum(predict == ground_truth)/len(ground_truth)
        
        return result
        

def main(args):
    file = open('baseline.txt', 'w')
    time_step_max = 5
    num_time_step_max = 5
    baseline_matrix = np.zeros((time_step_max + 1, num_time_step_max + 1))
    file1 = open('baseline_matrix', 'wb')
    for time_step in range(1,time_step_max + 1, 1):
        for num_time_step in range(1,num_time_step_max + 1, 1):
            result = baseline(float(args[0]), float(args[1]), args[2] == 'T', time_step, num_time_step)
            message = 'time_step = ' + str(time_step) + 'min, num_time_step = ' + str(num_time_step) + ', correctness = ' + str(result) + '\n'
            file.write(message)
            print(message)
            baseline_matrix[time_step][num_time_step] = result
        temp = np.argmax(baseline_matrix, axis = 1)
        message = 'when time_step = ' + str(time_step) + 'min, local maximum at num_time_step = '  + str(temp[time_step]) + ', maximum is ' + str(baseline_matrix[time_step][temp[time_step]]) + '\n'
        file.write(message)
        print(message)
    maximum_index = np.argmax(baseline_matrix)
    print(maximum_index)

    row = maximum_index // (num_time_step_max + 1)
    col = maximum_index % (num_time_step_max + 1)
    print(row)
    print(col)
    message = 'global maximum at time_step = ' + str(row) + 'min, num_time_step = ' + str(col) + ', maximum is ' + str(baseline_matrix[row][col])
    print(message)
    file.write(message)
    file.close()
    pickle.dump(baseline_matrix, file1)
    print(baseline_matrix)
    file1.close()

            


if __name__ == '__main__':
    decision_ratio = sys.argv[1]
    split_ratio = sys.argv[2]
    multifile = sys.argv[3]
    time_step = sys.argv[4]
    num_time_step = sys.argv[5]

    args = [decision_ratio, split_ratio, multifile, time_step, num_time_step]

    main(args)