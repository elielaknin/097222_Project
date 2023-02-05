import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import json
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_name', type=str, help='video name to analyze')
args = parser.parse_args()

def main():
    video_name = args.video_name
    video_path = f'data/videos/{video_name}_top'
    csv_path = f'data/videos/{video_name}_predictions.csv'
    output_path = f'data/videos/{video_name}_top_video'
    json_path = f'data/gestures_definitions.json'

    os.makedirs(output_path, exist_ok=True)
    video_label_df = pd.read_csv(csv_path, index_col='frame')

    new_predicted_video_path = os.path.join(output_path, (video_name + '_predicted.avi'))

    f = open(json_path, )
    gesture_dict_inv = json.load(f)['gesture']
    gesture_dict = {v: k for k, v in gesture_dict_inv.items()}
    f.close()

    calculate_confusion_matrix(video_label_df['gt'], video_label_df.iloc[:, 1:5], gesture_dict_inv, f"{video_name}_BL",
                               output_path)
    calculate_confusion_matrix(video_label_df['gt'], video_label_df.iloc[:, 5:9], gesture_dict_inv, f"{video_name}_ADV",
                               output_path)

    # save new video
    frame_width = int(640)
    frame_height = int(640)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(new_predicted_video_path, fourcc, 30, (frame_width, frame_height))

    number_of_frames = len(os.listdir(video_path))
    for frameNr, frame_name in enumerate(tqdm(sorted(os.listdir(video_path)))):

        frame = cv2.imread(os.path.join(video_path, frame_name))
        new_frame = np.full((frame_height, frame_width, 3), (255, 255, 255), dtype=np.uint8)
        new_frame[0:480, 0:640] = frame

        x_current_frame = 285
        cv2.arrowedLine(new_frame, (x_current_frame, 485),  (x_current_frame, 505), (0, 0, 255), 1)
        cv2.putText(new_frame, 'GT:', (20, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 3)
        cv2.putText(new_frame, 'BL-P:', (10, 565), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 3)
        cv2.putText(new_frame, 'ADV-P:', (1, 600), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 3)

        cv2.rectangle(new_frame, (60, 515), (509, 535), (0, 0, 0), 2)
        cv2.rectangle(new_frame, (60, 550), (509, 570), (0, 0, 0), 2)
        cv2.rectangle(new_frame, (60, 585), (509, 605), (0, 0, 0), 2)

        for idx in range(-225, 225):
            if idx+int(frameNr) < 0 or idx+int(frameNr) > number_of_frames:
                gt_color = (255, 255, 255)
                predicted_bl_color = (255, 255, 255)
                predicted_adv_color = (255, 255, 255)
            elif idx == 0:
                # red color
                gt_color = (0, 0, 255)
                gt_label = video_label_df['gt'].iloc[idx+int(frameNr)]
                predicted_bl_color = (0, 0, 255)
                predicted_bl_label = video_label_df['Baseline using 100% of train data'].iloc[idx+int(frameNr)]
                predicted_adv_color = (0, 0, 255)
                predicted_adv_label = video_label_df['Advanced using 100% of train data'].iloc[idx+int(frameNr)]
            else:
                gt_label = video_label_df['gt'].iloc[idx+int(frameNr)]
                gt_color = get_color(gt_label)
                predicted_bl_label = video_label_df['Baseline using 100% of train data'].iloc[idx+int(frameNr)]
                predicted_bl_color = get_color(predicted_bl_label)
                predicted_adv_label = video_label_df['Advanced using 100% of train data'].iloc[idx+int(frameNr)]
                predicted_adv_color = get_color(predicted_adv_label)

            if idx == 0:
                cv2.putText(new_frame, gesture_dict[f"G{gt_label}"], (512, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),
                            1, 1)
                cv2.putText(new_frame, gesture_dict[f"G{predicted_bl_label}"], (512, 565), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 1, 1)
                cv2.putText(new_frame, gesture_dict[f"G{predicted_adv_label}"], (512, 600), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 1, 1)

            cv2.line(new_frame, (x_current_frame+idx, 515), (x_current_frame+idx, 535), gt_color, 1)
            cv2.line(new_frame, (x_current_frame+idx, 550), (x_current_frame+idx, 570), predicted_bl_color, 1)
            cv2.line(new_frame, (x_current_frame + idx, 585), (x_current_frame + idx, 605), predicted_adv_color, 1)

        out.write(new_frame)
    out.release()


def get_color(label):
    if label == 0:
        return (255, 0, 0)
    elif label == 1:
        return (255, 255, 0)
    elif label == 2:
        return (0, 255, 0)
    elif label == 3:
        return (100, 50, 200)
    elif label == 4:
        return (255, 100, 200)
    elif label == 5:
        return (128, 128, 128)
    else:
        return (0, 0, 0)


def calculate_confusion_matrix(gt , video_label_df, gesture_dict_inv, video_name, output_path):
    axis_labels = list(gesture_dict_inv.keys())  # labels for axis

    fig = plt.figure(figsize=(20, 15))
    for idx in range(4):
        plt.subplot(2, 2, idx+1)
        name = video_label_df.columns[idx]
        plt.title(name, fontsize=20)
        cf_matrix = confusion_matrix(gt, video_label_df[name])
        sns.heatmap(cf_matrix, annot=True, fmt='g', xticklabels=axis_labels, yticklabels=axis_labels)
        plt.ylabel('Ground Truth', fontsize=15)
        plt.xlabel('Predicted', fontsize=15)
        plt.xticks(rotation=30)
    plt.tight_layout()

    cm_path = os.path.join(output_path, f'{video_name}_confusion_matrix.png')
    plt.savefig(cm_path)


if __name__ == "__main__":
    results = main()