import argparse
import pdb
import os
import math
from tqdm import tqdm
import pandas as pd
import numpy as np

def main():
    left_tool_path = '../../data/transcriptions_tools_left_new'
    right_tool_path = '../../data/transcriptions_tools_right_new'
    new_features_folder = '../../data/new_tool_usage_features'

    os.makedirs(new_features_folder, exist_ok=True)

    video_list = os.listdir(left_tool_path)

    for video_txt in tqdm(video_list):
        left_df = open_video_txt(os.path.join(left_tool_path, video_txt))
        right_df = open_video_txt(os.path.join(right_tool_path, video_txt))

        video_len = left_df['end'].iloc[-1]
        positional_encoding_matrix = get_positional_encoding(video_len)

        left_arr = np.zeros(video_len)
        for idx, row in left_df.iterrows():
            left_arr[row['start']:row['end']] = row['label']
        left_df = check_and_complete_df(pd.get_dummies(left_arr))
        left_label_and_post = np.concatenate([left_df.to_numpy(), positional_encoding_matrix], axis=1)


        right_arr = np.zeros(video_len)
        for idx, row in right_df.iterrows():
            right_arr[row['start']:row['end']] = row['label']
        right_df = check_and_complete_df(pd.get_dummies(right_arr))
        right_label_and_post = np.concatenate([right_df.to_numpy(), positional_encoding_matrix], axis=1)

        new_tool_pos_features = np.concatenate([left_label_and_post, right_label_and_post], axis=1).T
        video_name = video_txt.split('.txt')[0]
        np.save(os.path.join(new_features_folder, (video_name + '.npy')), new_tool_pos_features)


def check_and_complete_df(df):
    if df.shape[1] == 4:
        return df
    else:
        for idx in range(4):
            if idx not in df.columns:
                df[idx] = 0
        df = df[[0, 1, 2, 3]]
        return df


def get_positional_encoding(seq_len, output_embedding_space=16, n=10000):
    P = np.zeros((seq_len, output_embedding_space))
    for k in range(seq_len):
        for i in np.arange(int(output_embedding_space/2)):
            denominator = np.power(n, 2*i/output_embedding_space)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


def open_video_txt(path):
    file_ptr = open(path, 'r')
    label_content = file_ptr.read().split('\n')[:-1]
    file_ptr.close()

    time_list = []
    for line in label_content:
        start_time = int(line.split(' ')[0])
        end_time = int(line.split(' ')[1])
        label_time =int(line.split(' ')[2].split('T')[1])
        time_list.append((start_time, end_time, label_time))

    df = pd.DataFrame(time_list, columns=['start', 'end', 'label'])

    return df

if __name__ == "__main__":
    results = main()