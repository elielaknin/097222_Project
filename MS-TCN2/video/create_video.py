import argparse
import pdb
import os
import math

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import json

def main():
    video_path = '/Users/eliel/PycharmProjects/097222_Project/data/frames/P016_balloon1_top'
    csv_path = '/Users/eliel/PycharmProjects/097222_Project/data/frames/P016_balloon1_exp21_test_gt_predicted.csv'
    output_path = '/Users/eliel/PycharmProjects/097222_Project/data/frames/P016_balloon1_top_video'
    json_path = '/Users/eliel/PycharmProjects/097222_Project/data/gestures_definitions.json'

    os.makedirs(output_path, exist_ok=True)
    video_label_df = pd.read_csv(csv_path, index_col='Unnamed: 0')
    video_name = 'P016_balloon1'

    new_predicted_video_path = os.path.join(output_path, (video_name + '_predicted.avi'))

    f = open(json_path, )
    gesture_dict_inv = json.load(f)['gesture']
    gesture_dict = {v: k for k, v in gesture_dict_inv.items()}
    f.close()

    # save new video
    frame_width = int(640)
    frame_height = int(580)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(new_predicted_video_path, fourcc, 30, (frame_width, frame_height))

    number_of_frames = len(os.listdir(video_path))
    for frameNr, frame_name in enumerate(tqdm(sorted(os.listdir(video_path)))):

        frame = cv2.imread(os.path.join(video_path, frame_name))
        new_frame = np.full((580, 640, 3), (255, 255, 255), dtype=np.uint8)
        new_frame[0:480, 0:640] = frame

        x_current_frame = 275
        cv2.arrowedLine(new_frame, (x_current_frame, 485),  (x_current_frame, 505), (0, 0, 255), 1)
        cv2.putText(new_frame, 'GT:', (15, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 3)
        cv2.putText(new_frame, 'P:', (15, 565), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 3)

        cv2.rectangle(new_frame, (50, 515), (499, 535), (0, 0, 0), 2)
        cv2.rectangle(new_frame, (50, 550), (499, 570), (0, 0, 0), 2)

        for idx in range(-225, 225):
            if idx+int(frameNr) < 0 or idx+int(frameNr) > number_of_frames:
                gt_color = (255, 255, 255)
                predicted_color = (255, 255, 255)
            elif idx == 0:
                # red color
                gt_color = (0, 0, 255)
                gt_label = video_label_df['gt'].iloc[idx+int(frameNr)]
                predicted_color = (0, 0, 255)
                predicted_label = video_label_df['predicted'].iloc[idx+int(frameNr)]
            else:
                gt_label = video_label_df['gt'].iloc[idx+int(frameNr)]
                gt_color = get_color(gt_label)
                predicted_label = video_label_df['predicted'].iloc[idx+int(frameNr)]
                predicted_color = get_color(predicted_label)

            if idx == 0:
                cv2.putText(new_frame, gesture_dict[f"G{gt_label}"], (505, 530), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0),
                            1, 1)
                cv2.putText(new_frame, gesture_dict[f"G{predicted_label}"], (505, 565), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 1, 1)

            cv2.line(new_frame, (x_current_frame+idx, 515), (x_current_frame+idx, 535), gt_color, 1)
            cv2.line(new_frame, (x_current_frame+idx, 550), (x_current_frame+idx, 570), predicted_color, 1)


        # Displaying the image
        # cv2.imshow('image', new_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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



if __name__ == "__main__":
    results = main()