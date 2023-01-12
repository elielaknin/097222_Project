
import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="APAS")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='1280', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_folds_list_file = "../data/new_folds"
features_path = "../data/features/"
gt_path = "../data/transcriptions_gestures/"

mapping_file = "../data/mapping_gestures.txt"

model_dir = "../models/split_"+args.split
results_dir = "../results/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
for fold in range(5):
    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split)
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        batch_gen.read_data(os.path.join(vid_folds_list_file, f'train{fold}.txt'))
        trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=5, learning_rate=lr, device=device)



if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_folds_list_file, num_epochs, actions_dict, device, sample_rate)


'''
Alive test working  train acc of 0.372

TODO:
Phase 1:
- Fix loss train (nan)
- Support of cross validation
- New dataloader from train and val
- val loss and acc
- prediction of test set
- Implementation of B&W
- Add relevant metrics (accuracy, Edit distance, f1@k)
- Check framerate is correct
- Refactor code (like NLP course)

Phase 2:
- add module of Hw1
- Add FC layers using output of mstcn and hw1 prediction
- Support of factor of training dataset

Optional:
- use feature extraction of pre-trained Resnet50

'''