import argparse
import pdb
import os
import math

import pandas as pd
import numpy as np

def main():
    fold_path = '../../data/folds'
    new_fold_path = '../../data/new_folds'
    gt_path = '../../data/transcriptions_gestures'

    os.makedirs(new_fold_path, exist_ok=True)

    video_name = os.listdir(gt_path)
    video_name_list = [item.split('.')[0] for item in video_name]

    for fold_number in range(5):
        old_val_path = f"{fold_path}/valid{fold_number}.txt"
        file_ptr = open(old_val_path, 'r')
        old_val_content = file_ptr.read().split('\n')[:-1]
        new_val_content = [item.split('.')[0] for item in old_val_content]
        file_ptr.close()

        old_test_path = f"{fold_path}/test{fold_number}.txt"
        file_ptr = open(old_test_path, 'r')
        old_test_content = file_ptr.read().split('\n')[:-1]
        new_test_content = [item.split('.')[0] for item in old_test_content]
        file_ptr.close()

        new_train_content = []
        for item in video_name_list:
            if item in new_val_content or item in new_test_content:
                continue
            else:
                new_train_content.append(item)

        # save new train
        new_path = os.path.join(new_fold_path, f"train{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_train_content:
                f.write("%s\n" % line)

        # save new valid
        new_path = os.path.join(new_fold_path, f"valid{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_val_content:
                f.write("%s\n" % line)

        # save new test
        new_path = os.path.join(new_fold_path, f"test{fold_number}.txt")
        with open(new_path, 'w') as f:
            for line in new_test_content:
                f.write("%s\n" % line)

        print('a')




if __name__ == "__main__":
    results = main()