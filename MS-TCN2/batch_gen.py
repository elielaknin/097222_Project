
import torch
import numpy as np
import random
import os


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, run_local, data_ratio_to_use=1):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.run_local = run_local
        self.data_ratio_to_use = data_ratio_to_use


    def get_video_name(self):
        return self.list_of_examples[self.index-1]

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        if self.data_ratio_to_use == 1:
            random.shuffle(self.list_of_examples)
        else:
            number_of_sample_to_use = int(len(self.list_of_examples)*self.data_ratio_to_use)
            random.shuffle(self.list_of_examples)
            self.list_of_examples = self.list_of_examples[0:number_of_sample_to_use]



    def next_batch(self, batch_size, current_fold_number):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            # print(f"current video is {vid}")
            print(f"self.run_local is {self.run_local}")
            if self.run_local:
                features = np.load(self.features_path + vid + '.npy')
            else:
                feature_path_with_fold = os.path.join(self.features_path, f"fold_{current_fold_number}/")
                print(feature_path_with_fold)
                features = np.load(feature_path_with_fold + vid + '.npy')

            file_ptr = open(self.gt_path + vid + '.txt', 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes_list = []
            for i in range(len(content)):
                start = int(content[i].split(' ')[0])
                end = int(content[i].split(' ')[1])
                label = self.actions_dict[content[i].split(' ')[2]]
                if end-start <= 0:
                    continue
                classes_list.append(label*np.ones(end-start+1))

            classes = np.concatenate(classes_list, axis=0)
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
