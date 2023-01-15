import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def train_full(train_dataloader, test_dataloader, num_epochs, model, device, optimizer, model_name, acum_grad_steps,
               my_name, scheduler=None):
    train_loss_list_epochs = []
    train_uas_list_epochs = []

    test_loss_list_epochs = []
    test_uas_list_epochs = []
    max_uas = 0

    for epoch_idx in range(num_epochs):

        # train
        print("\nStart epoch {} / {}".format(epoch_idx + 1, num_epochs))
        train_loss, train_uas = train_one_epoch(optimizer, device, model, acum_grad_steps, train_dataloader)
        print("Train results: epoch {}\tloss = {}\t: UAS = {}%".format(epoch_idx + 1, train_loss, train_uas))
        train_loss_list_epochs.append(train_loss)
        train_uas_list_epochs.append(train_uas)

        # test
        test_loss, test_uas = test_one_epoch(device, model, test_dataloader)
        print("Test results: epoch {}\tloss = {}\t: UAS = {}%".format(epoch_idx + 1, test_loss, test_uas))
        test_loss_list_epochs.append(test_loss)
        test_uas_list_epochs.append(test_uas)

        if scheduler is not None:
            scheduler.step()
            # scheduler.step(min(test_loss_list_epochs))

        # update model
        if test_uas > max_uas:
            max_uas = test_uas
            save_model_name = my_name + model_name
            torch.save(model.state_dict(), f"{save_model_name}.pt")
            print("Saving new best model with uas of ", max_uas)

        df = pd.DataFrame(list(zip(test_uas_list_epochs, train_uas_list_epochs, test_loss_list_epochs,
                                   train_loss_list_epochs)),
                          columns=['UAS_test', 'UAS_train', 'Loss_test', 'Loss_train'])
        pd_name = my_name + '_model_run_info.csv'
        df.to_csv(pd_name)

    # return for graphs
    return test_uas_list_epochs, train_uas_list_epochs, test_loss_list_epochs, train_loss_list_epochs


def train_one_epoch(optimizer, device, model, acum_grad_steps, dl, sentence_drop_out=0, words_drop_out=0, mask_ind=0):
    train_loss_list_batches = []
    train_uas_list_batches = []

    print("learning rate is:", optimizer.param_groups[0]['lr'])

    i = 0
    for data_batch in tqdm(dl):

        sentence_info_list = [torch.clone(x).squeeze().to(device) for x in data_batch]
        sentence_len = sentence_info_list[2].item()
        gt_tree = sentence_info_list[3]

        # dropout masks
        to_dropout_sentence = (np.random.rand(1)[0] < sentence_drop_out)
        if to_dropout_sentence:
            masks_inds = np.random.rand(sentence_len) < words_drop_out
            sentence_info_list[0][masks_inds] = mask_ind

        scores_arr = model.forward(sentence_info_list[0:3]).to(device)

        # Calculate the loss
        loss = nll_loss(scores_arr, gt_tree, sentence_len)
        loss = loss / acum_grad_steps
        loss.backward()
        train_loss_list_batches.append(loss.item())

        if i % acum_grad_steps == 0:
            optimizer.step()
            model.zero_grad()

            predicted_tree, _ = decode_mst(energy=scores_arr.cpu().detach(), length=scores_arr.shape[0], has_labels=False)
            uas_score = calc_uas(predicted_tree, gt_tree)
            train_uas_list_batches.append(uas_score)

        i += 1


    printable_loss = acum_grad_steps * np.sum(train_loss_list_batches) / len(train_loss_list_batches)
    return printable_loss, np.mean(train_uas_list_batches)


def test_one_epoch(device, model, dl):
    test_loss_list_batches = []
    test_uas_list_batches = []

    for data_batch in tqdm(dl):

        sentence_info_list = [x.squeeze().to(device) for x in data_batch]

        sentence_len = sentence_info_list[2].item()
        gt_tree = sentence_info_list[3]
        scores_arr = model.forward(sentence_info_list[0:3]).to(device)

        # Calculate the loss
        loss = nll_loss(scores_arr, gt_tree, sentence_len)

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
        predicted_tree, _ = decode_mst(energy=scores_arr.cpu().detach(), length=scores_arr.shape[0], has_labels=False)
        uas_score = calc_uas(predicted_tree, gt_tree)

        test_loss_list_batches.append(loss.item())
        test_uas_list_batches.append(uas_score)

    return np.mean(test_loss_list_batches), np.mean(test_uas_list_batches)
