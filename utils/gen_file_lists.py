import os
import sys
import math
from glob import glob
from general_utils import shuffle_list

# constant
TRAIN_FILE_LIST = './dataset/train.txt'
TEST_FILE_LIST = './dataset/test.txt'
EVALUATE_FILE_LIST = './dataset/evaluation.txt'

DATA_DIR = './dataset/chart/'
LABEL_DIR = './dataset/legend/'

# global variables
num_sample = 3840

data_extension = '*.csv'
label_extension = '*.png'


def generate_list_file():
	training_num = int(math.floor(num_sample * 0.8))
	evaluate_num = int(math.floor(num_sample * 0.1))
	test_num = num_sample - training_num - evaluate_num

	data_paths = sorted(glob(os.path.join(DATA_DIR, data_extension)))
	label_paths = sorted(glob(os.path.join(LABEL_DIR, label_extension)))
	data_paths, label_paths = shuffle_list(data_paths, label_paths)

	train = open(TRAIN_FILE_LIST, 'w')
	for i in range(training_num):
		print('{}\t{}'.format(data_paths[i], label_paths[i]), file=train)
	train.close()

	test = open(TEST_FILE_LIST, 'w')
	for i in range(test_num):
		print('{}\t{}'.format(data_paths[i + training_num], label_paths[i + training_num]), file=test)
	test.close()

	evaluate = open(EVALUATE_FILE_LIST,'w')
	for i in range(evaluate_num):
		print('{}\t{}'.format(data_paths[i + training_num + test_num], label_paths[i + training_num + test_num]), file=evaluate)
	evaluate.close()

if __name__ == '__main__':
	num_sample = int(sys.argv[1])

	data_extension = sys.argv[2]  # *.png   *.csv  (without ' ')
	label_extension = sys.argv[3]  # *.png   *.csv

	generate_list_file()