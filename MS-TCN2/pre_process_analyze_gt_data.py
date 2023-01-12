import argparse
import pdb
import os
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def main():
    result_path = '../results/GT_label_data_stats'
    gt_path = '../data/transcriptions_gestures'
    gt_definition = '../data/gestures_definitions.json'

    f = open(gt_definition)
    gesture_label_dict = json.load(f)['gesture']
    gesture_label_dict_inv = dict(zip(gesture_label_dict.values(), gesture_label_dict.keys()))

    os.makedirs(result_path, exist_ok=True)

    video_list = os.listdir(gt_path)

    video_counter_stats_dict = {}
    video_appearance_stats_dict = {}
    for video_label in video_list:
        label_path = os.path.join(gt_path,f"{video_label}")
        file_ptr = open(label_path, 'r')
        label_content = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        time_cnt = {"G0": 0, "G1": 0, "G2": 0, "G3": 0, "G4": 0, "G5": 0}
        appearance_cnt = {"G0": 0, "G1": 0, "G2": 0, "G3": 0, "G4": 0, "G5": 0}
        for line in label_content:
            start_time = int(line.split(' ')[0])
            end_time = int(line.split(' ')[1])
            label_time = line.split(' ')[2]

            delta_time = end_time-start_time
            if delta_time > 0:
                appearance_cnt[label_time] += 1
                time_cnt[label_time] += delta_time
        video_counter_stats_dict[video_label.split('.txt')[0]] = time_cnt
        video_appearance_stats_dict[video_label.split('.txt')[0]] = appearance_cnt


    video_counter_df = pd.DataFrame(video_counter_stats_dict, dtype='float')
    video_appearance_df = pd.DataFrame(video_appearance_stats_dict, dtype='float')

    video_counter_df = video_counter_df / video_counter_df.sum(axis=0)
    video_appearance_df = video_appearance_df / video_appearance_df.sum(axis=0)

    video_counter_df.rename(gesture_label_dict_inv, inplace=True)
    video_counter_df = video_counter_df.T
    video_appearance_df.rename(gesture_label_dict_inv, inplace=True)
    video_appearance_df = video_appearance_df.T

    pos = [1, 2, 3, 4, 5, 6]
    # labels = ['a', 'b', 'c', 'd', 'e', 'f']
    labels = list(gesture_label_dict_inv.values())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 12))
    ax.violinplot(video_counter_df, pos)
    ax.set_xticklabels(labels)
    plt.title('Propability of time of each label per video')
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(result_path, 'counter_delta_time_propa_violin_plot.png'))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 12))
    ax.violinplot(video_appearance_df, pos)
    ax.set_xticklabels(labels)
    plt.title('Propability of appearance of each label per video')
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(result_path, 'number_of_appearance_propa_violin_plot.png'))

    print('a')




if __name__ == "__main__":
    results = main()