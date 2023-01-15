import json
import os.path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
import eval
from tqdm import tqdm
import wandb
import itertools


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, model_type, hp, dim, num_classes, dataset, output_folder):
        if model_type == 'baseline':
            self.model = MS_TCN2(hp['num_layers_PG'], hp['num_layers_R'], hp['num_R'], hp['num_f_maps'], dim, num_classes)
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
            self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.output_folder = output_folder

        logger.add(os.path.join(output_folder, dataset + "_{time}.log"))
        logger.add(sys.stdout, colorize=True, format="{message}")


    def full_train(self, save_dir, train_dl, val_dl, test_dl, fold_number, hp, num_epochs, device,
                   exp_name, exp_idx):

        #save HP dict
        with open(os.path.join(save_dir, "HP.json"), "w") as write_file:
            json.dump(hp, write_file, indent=2)

        self.model.train()
        self.model.to(device)

        exp_hp_number = f"exp_{exp_idx}"
        wandb.init(project=f"{exp_name}_fold_{fold_number}", name=exp_hp_number, config=hp)

        # wandb.run.name = exp_hp_number
        # wandb.config.update(hp)

        min_val_loss = np.inf
        max_val_acc, max_val_edit, max_val_f1at10, max_val_f1at25, max_val_f1at50 = 0, 0, 0, 0, 0
        best_epoch = -1

        optimizer = optim.Adam(self.model.parameters(), lr=hp['learning_rate'])
        epochs_results_list = []

        for epoch in tqdm(range(num_epochs)):

            # train part
            loss_train, acc_train = self.train(device, optimizer, train_dl, epoch, hp['batch_size'], fold_number)

            # val part
            loss_val, acc_val, edit_score_val, f10_val, f25_val, f50_val = self.val(device, val_dl, epoch, fold_number)

            epochs_results_list.append((epoch, loss_train, acc_train, loss_val, acc_val, edit_score_val, f10_val, f25_val, f50_val))
            wandb.log({"loss_train": loss_train, "acc_train": acc_train, "loss_val": loss_val, "acc_val": acc_val,
                       "edit_score_val": edit_score_val, "f10_val": f10_val, "f25_val": f25_val, "f50_val": f50_val})

            # Save best validation model
            if loss_val < min_val_loss:
                min_val_loss = loss_val
                max_val_acc, max_val_edit = acc_val, edit_score_val
                max_val_f1at10, max_val_f1at25, max_val_f1at50 = f10_val, f25_val, f50_val
                best_epoch = epoch

                save_model_name = f"best_model"
                torch.save(self.model.state_dict(), os.path.join(save_dir, f"{save_model_name}.pth"))
                # print('Save new best model')

        wandb.finish(quiet=True)
        pd.DataFrame(epochs_results_list, columns=['epochs', 'loss_train', 'acc_train', 'loss_val', 'acc_val',
                                                   'edit_score_val','f10_val', 'f25_val', 'f50_val']).to_csv(
            os.path.join(save_dir, 'run_summary.csv'), index=False)

        # Test part on the best model
        test_metrics_df = self.test(device, test_dl, save_dir, save_model_name, fold_number)
        test_metrics_df.to_csv(os.path.join(save_dir, 'test_metrics_df.csv'))

        # logger.complete()

        return (f"exp_idx_{exp_idx}", best_epoch, min_val_loss, max_val_acc, max_val_edit, max_val_f1at10, max_val_f1at25,
                max_val_f1at50, test_metrics_df['loss'].mean(), test_metrics_df['accuracy'].mean(),
                test_metrics_df['Edit distance'].mean(), test_metrics_df['f1@10'].mean(),
                test_metrics_df['f1@25'].mean(), test_metrics_df['f1@50'].mean())

    """
    Train function of the model, for one epoch
    """
    def train(self, device, optimizer, train_dl, epoch, batch_size, current_fold_number):
            epoch_loss_train = 0
            correct_train = 0
            total_train = 0
            # train part
            self.model.train()
            while train_dl.has_next():
                batch_input, batch_target, mask = train_dl.next_batch(batch_size, current_fold_number)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                self.model.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(
                        torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                    min=0, max=16) * mask[:, :, 1:])

                epoch_loss_train += loss.item()
                # print(loss.item())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_train += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_train += torch.sum(mask[:, 0, :]).item()

            train_dl.reset()
            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            loss_train = epoch_loss_train / len(train_dl.list_of_examples)
            acc_train = 100 * float(correct_train) / total_train

            # logger.info("[epoch %d]: epoch loss = %f,  acc = %f" % (epoch + 1, loss_train, acc_train))

            return loss_train, acc_train


    """
    Val function of the model, for one epoch. After the training, process on the validation test to choose the 
    best hyper-params. 
    """
    def val(self, device, val_dl, epoch, current_fold_number, batch_size=1):
        epoch_loss_val = 0
        correct_val = 0
        total_val = 0
        total_edit_score_val, total_f10_val, total_f25_val, total_f50_val = 0, 0, 0, 0

        self.model.eval()
        with torch.no_grad():
            while val_dl.has_next():
                batch_input, batch_target, mask = val_dl.next_batch(batch_size, current_fold_number)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                self.model.to(device)
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss_val += loss.item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_val += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_val += torch.sum(mask[:, 0, :]).item()

                y_pred = np.squeeze(predicted.cpu().detach().numpy())
                y_gt = np.squeeze(batch_target.cpu().detach().numpy())

                f10, f25, f50 = eval.multiple_f_score(y_pred, y_gt)
                total_f10_val += f10
                total_f25_val += f25
                total_f50_val += f50

                # total_edit_score_val += eval.levenstein(y_pred, y_gt, True)
                total_edit_score_val = 0

        val_dl.reset()
        loss_val = epoch_loss_val / len(val_dl.list_of_examples)
        acc_val = 100 * float(correct_val) / total_val
        edit_score_val = total_edit_score_val / len(val_dl.list_of_examples)
        f10_val = total_f10_val / len(val_dl.list_of_examples)
        f25_val = total_f25_val / len(val_dl.list_of_examples)
        f50_val = total_f50_val / len(val_dl.list_of_examples)

        # logger.info("[epoch %d]: epoch loss = %f,   acc = %f, f1@10 = %f, f1@25 = %f, f1@50 = %f,"
        #             % (epoch + 1, loss_val, acc_val, f10_val, f25_val, f50_val))
        return loss_val, acc_val, edit_score_val, f10_val, f25_val, f50_val


    """
    Test function of the model. Using the model that achieved the average maximum accuracy between the 5 cross 
    validation on the validation set. Take the best models and run it on the test set.
    """
    def test(self, device, test_dl, save_dir, save_model_name, current_fold_number, batch_size=1):

        test_video_dir = os.path.join(save_dir, 'test_video')
        os.makedirs(test_video_dir, exist_ok=True)

        test_metrics_list = []
        # load best model saved
        self.model.load_state_dict(torch.load(os.path.join(save_dir, f"{save_model_name}.pth")))
        self.model.eval()
        with torch.no_grad():
            while test_dl.has_next():
                batch_input, batch_target, mask = test_dl.next_batch(batch_size, current_fold_number)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(
                    device)
                self.model.to(device)
                predictions = self.model(batch_input)
                video_name = test_dl.get_video_name()

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                    batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                   F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                          min=0, max=16) * mask[:, :, 1:])

                loss_test = loss.item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct_test = ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total_test = torch.sum(mask[:, 0, :]).item()
                acc_test = 100 * float(correct_test) / total_test

                y_pred = np.squeeze(predicted.cpu().detach().numpy())
                y_gt = np.squeeze(batch_target.cpu().detach().numpy())
                # edit_score_test = eval.levenstein(y_pred, y_gt, True)
                edit_score_test = 0
                f10, f25, f50 = eval.multiple_f_score(y_pred, y_gt)

                (pd.DataFrame([y_pred, y_gt], index=['predicted', 'gt']).T).to_csv(os.path.join(test_video_dir, f"{video_name}.csv"))

                test_metrics_list.append((video_name, acc_test, loss_test, edit_score_test, f10, f25, f50))
                # logger.info(f"Test on {video_name} - loss = {loss_test}, acc = {acc_test}, edit score = {edit_score_test}"
                #             f"f1@10 = {f10}, f1@25 = {f25}, f1@50 = {f50}")

        test_dl.reset()
        test_metrics_df = pd.DataFrame(test_metrics_list, columns=['video_name', 'accuracy', 'loss',
                                                                   'Edit distance', 'f1@10', 'f1@25', 'f1@50'])
        return test_metrics_df




    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

