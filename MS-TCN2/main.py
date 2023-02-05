import pandas as pd
import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cuda is {torch.cuda.is_available()}")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="APAS")
parser.add_argument('--train_ratio', default='1', type=float)
parser.add_argument('--run_local', action='store_true')
parser.add_argument('--model_type', default='baseline') #baseline or advanced
parser.add_argument('--active_wandb', action='store_true')

parser.add_argument('--features_dim', default='1280', type=int)
parser.add_argument('--bz', default='1', type=int, nargs="+")
parser.add_argument('--lr', default='0.0005', type=float, nargs="+")

parser.add_argument('--num_f_maps', default='64', type=int, nargs="+")

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int, nargs="+")
parser.add_argument('--num_layers_R', type=int, nargs="+")
parser.add_argument('--num_R', type=int, nargs="+")

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

sample_rate = 1

vid_folds_list_file = "../data/new_filter_folds"

features_path = "../data/features/"
gt_path = "../data/transcriptions_gestures/"
mapping_file = "../data/mapping_gestures.txt"
tool_usage_features = "../data/new_tool_usage_features/"


exp_name = f"{args.model_type}_model_train_ratio_{args.train_ratio}"
model_dir = f"../exp/{exp_name}/"
# results_dir = f"../exp/{args.exp_name}/results/"

# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

hp_dict = {
    "num_layers_PG": num_layers_PG,
    "num_layers_R": num_layers_R,
    "num_R": num_R,
    "num_f_maps": num_f_maps,
    "batch_size": bz,
    "learning_rate": lr
}

if args.model_type == 'advanced':
    features_dim = features_dim + 40

for fold in range(5):
    if args.action == "train":

        results_list = []

        keys, values = zip(*hp_dict.items())
        hp_combi_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # new model path per fold
        model_dir_fold = os.path.join(model_dir, f"fold_{fold}")
        os.makedirs(model_dir_fold, exist_ok=True)

        #dataloader creation
        train_dl = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, args.run_local,
                                  tool_usage_features, args.model_type, args.train_ratio)
        train_dl.read_data(os.path.join(vid_folds_list_file, f'train{fold}.txt'))

        val_dl = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, args.run_local,
                                tool_usage_features, args.model_type)
        val_dl.read_data(os.path.join(vid_folds_list_file, f'valid{fold}.txt'))

        test_dl = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, args.run_local,
                                 tool_usage_features, args.model_type)
        test_dl.read_data(os.path.join(vid_folds_list_file, f'test{fold}.txt'))

        total_videos_number = len(train_dl.list_of_examples) + len(val_dl.list_of_examples) + len(
            test_dl.list_of_examples)
        print(f"Folder #{fold} - number of video used: Train {len(train_dl.list_of_examples)} videos, "
              f"Val {len(val_dl.list_of_examples)} videos, Test {len(test_dl.list_of_examples)} videos"
              f"- Ratio {int(100 * len(train_dl.list_of_examples) / total_videos_number)}/"
              f"{int(100 * len(val_dl.list_of_examples) / total_videos_number)}/"
              f"{int(100 * len(test_dl.list_of_examples) / total_videos_number)}")

        for exp_idx, hp in enumerate(hp_combi_list):
            print(f"{exp_idx+1}/{len(hp_combi_list)} - HP: {hp}")

            exp_model_dir_fold = os.path.join(model_dir_fold, f"exp_{exp_idx}")
            os.makedirs(exp_model_dir_fold, exist_ok=True)
            # check if csv summary file already exist:
            if os.path.exists(os.path.join(exp_model_dir_fold, 'run_summary.csv')):
                print("Configuration was already done, continue to next..")
                continue

            trainer = Trainer(args.model_type, hp, features_dim, num_classes, args.dataset, exp_model_dir_fold)

            results_tuple = trainer.full_train(exp_model_dir_fold, train_dl, val_dl, test_dl, fold, hp, num_epochs,
                                               device, exp_name, exp_idx, args.active_wandb)
            results_list.append(results_tuple)

        results_df = pd.DataFrame(results_list, columns=['exp_name', 'best_epoch', 'val_loss', 'val_acc', 'val_edit',
                                                         'val_f1@25', 'val_f1@50', 'val_f1@10', 'test_loss', 'test_acc',
                                                         'test_edit', 'test_f1@25', 'test_f1@50', 'test_f1@10'])
        results_df.to_csv(os.path.join(model_dir_fold, 'summary_results.csv'))