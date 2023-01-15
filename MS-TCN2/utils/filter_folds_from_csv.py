import argparse
import pdb
import os
import math

import pandas as pd
import numpy as np

def main():
    original_folds_path = '../../data/new_folds'
    filter_folds_path = '../../data/new_filter_folds'
    csv_path = '../../data/new_video_list_n94.csv'

    os.makedirs(filter_folds_path, exist_ok=True)

    csv_list_filter = list(pd.read_csv(csv_path, header=None)[0])

    for fold_number in range(5):

        old_val_path = f"{original_folds_path}/train{fold_number}.txt"
        file_ptr = open(old_val_path, 'r')
        old_train_content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        new_train_content = []
        for item in old_train_content:
            if item in csv_list_filter:
                new_train_content.append(item)


        old_val_path = f"{original_folds_path}/valid{fold_number}.txt"
        file_ptr = open(old_val_path, 'r')
        old_val_content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        new_val_content = []
        for item in old_val_content:
            if item in csv_list_filter:
                new_val_content.append(item)


        old_test_path = f"{original_folds_path}/test{fold_number}.txt"
        file_ptr = open(old_test_path, 'r')
        old_test_content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        new_test_content = []
        for item in old_test_content:
            if item in csv_list_filter:
                new_test_content.append(item)

        # save new train
        new_path = os.path.join(filter_folds_path, f"train{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_train_content:
                f.write("%s\n" % line)

        # save new valid
        new_path = os.path.join(filter_folds_path, f"valid{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_val_content:
                f.write("%s\n" % line)

        # save new test
        new_path = os.path.join(filter_folds_path, f"test{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_test_content:
                f.write("%s\n" % line)

        print(f"for fold number {fold_number} - number of files -  train: {len(new_train_content)}, "
              f"val: {len(new_val_content)}, test: {len(new_test_content)}")




if __name__ == "__main__":
    results = main()