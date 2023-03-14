import argparse
import collections
import json
import os
from glob import glob

from tqdm import tqdm

from utils.cmrc2018_evaluate import get_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data dir
    parser.add_argument('--test_file', type=str, default='cmrc2018_test_2k.json')
    parser.add_argument('--predict_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='predictions_test.json')

    # use some global vars for convenience
    args = parser.parse_args()

    eval_result = get_eval(args.test_file, args.predict_file)

    print(eval_result)

    json.dump(eval_result, open(args.output_file, "w"))
    
