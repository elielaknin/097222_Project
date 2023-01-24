import argparse
import pdb
import os
import math

import pandas as pd
import numpy as np
import eval
from sklearn.metrics import accuracy_score
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--results_folder_path', default='exp')
parser.add_argument('--model_type', default='baseline')
parser.add_argument('--train_ratio', type=float, nargs="+")
args = parser.parse_args()

def main():

    results_folder_path = args.results_folder_path
    model_type = args.model_type
    ratio_list = list(args.train_ratio)

    #build_path
    for ratio in ratio_list:
        folder_path = os.path.join(results_folder_path, f"{model_type}_model_train_ratio_{str(ratio)}")

        csv_path = os.path.join(results_folder_path, f"test_summary_metrics_result_for_{model_type}_model_train_ratio_{ratio}.csv")
        test_video_metrics = {}
        for fold_number in range(5):
            print(f"{fold_number+1}/5")
            fold_folder_path = os.path.join(folder_path, f"fold_{fold_number}")

            #open summary csv
            summary_cv = pd.read_csv(os.path.join(fold_folder_path, 'summary_results.csv'), index_col='Unnamed: 0')
            summary_cv.sort_values('val_acc', ascending=False, inplace=True)
            best_exp_idx = summary_cv['exp_name'].values[0].split('idx_')[1]

            #enter best exp from val result
            best_exp_test_videos = os.path.join(os.path.join(fold_folder_path, f"exp_{best_exp_idx}", 'test_video'))
            for video in tqdm(os.listdir(best_exp_test_videos)):
                #open video csv
                video_df = pd.read_csv(os.path.join(best_exp_test_videos, video), index_col='Unnamed: 0')

                accuracy = accuracy_score(video_df['gt'], video_df['predicted'])
                f10, f25, f50 = eval.multiple_f_score(video_df['predicted'], video_df['gt'])
                edit_score = eval.levenstein(video_df['predicted'], video_df['gt'], True)

                test_video_metrics[video.split('.csv')[0]] = (accuracy, edit_score, f10, f25, f50)


        res_df = pd.DataFrame(test_video_metrics, index=['Accuracy', 'Edit distance score', 'f1@10', 'f1@25', 'f1@50']).T
        res_df.index.name = 'video'
        res_df.to_csv(csv_path)
        print('Done!')






if __name__ == "__main__":
    results = main()