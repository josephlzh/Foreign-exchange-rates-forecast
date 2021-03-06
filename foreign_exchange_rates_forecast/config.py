GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 70000
ENCODER_HIDDEN_SIZE = 64
DECODER_HIDDEN_SIZE = 64
SPLIT_RATIO = 0.9
DRIVING = 'stocks/MSFT.US.csv'
TARGET = 'stocks/AAPL.US.csv'
DATA_DIR = 'FX/'
TIME_STEP = 20
BINARY_DATASET_HEADER = 'processed_data/header_EURUSD_30min'
BINARY_DATASET_DIR = 'processed_data/shuffled_EURUSD_30min_part_'
GRANULARITY = [30, 60, 480, 1440, 2880, 10080, 20160]
DROP_OUT = 0
B = 1
MAX_SINGLE_FILE_LINE_NUM = 16000
VALIDATION_LINE_NUM = 1000
BATCH_SIZE = 32
MEMORY_CAPACITY = 100000
ACTION_NUM = 3
TARGET_UPDATE = 50