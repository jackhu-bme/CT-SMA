import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

import scipy.stats as stats

from functools import partial

from bfe_predict import *

from predict_analysis import ct_patch_norm

import argparse


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, ct_patch_dir, bfe_patch_dir, txt_path, morphology_min, morphology_max):
        self.bfe_patch_dir = bfe_patch_dir
        self.ct_patch_dir = ct_patch_dir
        with open(txt_path, "r") as f:
            self.txt_list = f.readlines()
        self.bfe_list = [os.path.join(self.bfe_patch_dir, txt_name.strip().split('/')[-1]) for txt_name in
                         self.txt_list]
        self.ct_list = [os.path.join(self.ct_patch_dir, txt_name.strip().split('/')[-1]) for txt_name in
                         self.txt_list]
        self.morphology_min = morphology_min
        self.morphology_max = morphology_max

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        ct_patch_path = self.ct_list[idx]
        bfe_patch_path = self.bfe_list[idx]
        ct_patch = np.load(ct_patch_path, allow_pickle=True)
        bfe_patch = np.load(bfe_patch_path, allow_pickle=True)
        try:
            bfe_patch_img = bfe_patch_norm(bfe_patch["patch"])
        except Exception:
            print(f"warning!wrong file:{bfe_patch_path}")
            raise ValueError
        bfe_mask_img = bfe_patch["mask"]
        # print("bfe morph params:{}".format(bfe_patch["morph_params"]))
        try:
            bfe_morph_params = torch.tensor(
                bfe_morphology_norm(bfe_patch["morph_params"], self.morphology_min, self.morphology_max)).float()
        except Exception:
            print(f"warning!wrong file:{bfe_patch_path}")
            raise ValueError
        bfe_input = torch.tensor(np.stack([bfe_patch_img, bfe_mask_img], axis=0)).float()
        ct_path_img = ct_patch_norm(ct_patch["patch"])
        ct_patch_mask = ct_patch["mask"]
        ct_input = torch.tensor(np.stack([ct_path_img, ct_patch_mask], axis=0)).float()
        return (ct_input, bfe_input), bfe_morph_params


def distill_train(opt, exp_predix, txt_dir, save_dir, norm_dir, bfe_patch_dir, ct_patch_dir, epochs):
    batch_size = 4
    use_wandb = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0003
    gamma_list = [float(num) for num in (opt.gamma_list[1:-1].replace(" ", "")).split(",")]
    keep_feature_list = [int(num) for num in (opt.keep_feature_list[1:-1].replace(" ", "")).split(",")]
    use_pred = False
    assert len(gamma_list) == len(keep_feature_list), f"gamma_list:{gamma_list}, keep_feature_list:{keep_feature_list}"
    exp_name = f"revised_try8_distill_keep_feature_list:{keep_feature_list}_gamma:{gamma_list}_batch{batch_size}_lr{learning_rate}_pretrained"
    if exp_predix:
        exp_name = exp_predix + exp_name
    if use_wandb:
        wandb.init(project="distilled_ct_predict", name=exp_name, settings=wandb.Settings(_disable_stats=True))
    train_txt_path = txt_dir + "/train.txt"
    val_txt_path = txt_dir + "/val.txt"
    bfe_morph_min_array = np.load(os.path.join(norm_dir, "morph_min.npy"), allow_pickle=True)
    bfe_morph_max_array = np.load(os.path.join(norm_dir, "morph_max.npy"), allow_pickle=True)
    train_dataset = DistillDataset(ct_patch_dir, bfe_patch_dir, train_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    val_dataset = DistillDataset(ct_patch_dir, bfe_patch_dir, val_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    bfe_teacher = BFE_Predictor(input_channels=2, middle_channels=64, feature_channels=61, output_channels=4, type="res50_3d",
                                keep_feature_list=keep_feature_list, bfe_pretrained=True).to(device)
    bfe_teacher.to(device)
    bfe_teacher.eval()
    ct_student = CT_Predictor(input_channels=2, feature_channels=61, output_channels=4, type="res50", keep_feature_list=keep_feature_list, verbose=False)
    ct_student.to(device)
    if use_wandb:
        wandb.watch(ct_student)
    optimizer = torch.optim.Adam(ct_student.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.1)
    criterion_predict = nn.MSELoss()
    criterion_feature_map = nn.L1Loss()
    min_val_pred_loss = inf
    for epoch in tqdm(range(epochs)):
        train_predict_loss = 0
        train_feat_loss = 0
        val_predict_loss = 0
        val_feat_loss = 0
        if use_pred:
            train_ts_pred_loss = 0
            val_ts_pred_loss = 0
        for i, ((ct_input, bfe_input), bfe_morph_params) in enumerate(train_loader):
            bfe_input = bfe_input.to(device)
            ct_input = ct_input.to(device)
            bfe_morph_params = bfe_morph_params.to(device)
            optimizer.zero_grad()
            student_output = ct_student(ct_input)
            with torch.no_grad():
                teacher_output = bfe_teacher(bfe_input)
            if use_pred:
                ts_pred_loss = criterion_predict(teacher_output, student_output) * use_pred
            predict_loss = criterion_predict(student_output, bfe_morph_params)
            feature_difference_list = [criterion_feature_map(teacher_feat, student_feat) for (teacher_feat, student_feat) in
                                       zip(list(bfe_teacher.feature_dict.values()), list(ct_student.feature_dict.values()))]
            feature_loss = (torch.stack(feature_difference_list) * torch.tensor(gamma_list).to(device)).sum() if keep_feature_list else 0
            loss = predict_loss + feature_loss
            if use_pred:
                loss = loss + ts_pred_loss
            loss.backward()
            optimizer.step()
            train_predict_loss += predict_loss.item() / len(train_loader)
            if keep_feature_list:
                train_feat_loss += feature_loss.item()/len(train_loader)
            if use_pred:
                train_ts_pred_loss += ts_pred_loss.item()/len(train_loader)
        print(f"train_predict_loss: {train_predict_loss} for {i}th training!")
        print(f"train_feat_loss: {train_feat_loss}")
        scheduler.step()
        if use_wandb:
            train_log_dict = {"epoch": epoch+1, "train_pred_loss": train_predict_loss, "train_feature_loss": train_feat_loss,
                       "train_loss": train_predict_loss + train_feat_loss}
            if use_pred:
                train_log_dict["train_ts_pred_loss"] = train_ts_pred_loss
            wandb.log(train_log_dict)
        for i, ((ct_input, bfe_input), bfe_morph_params) in enumerate(val_loader):
            bfe_input = bfe_input.to(device)
            ct_input = ct_input.to(device)
            bfe_morph_params = bfe_morph_params.to(device)
            optimizer.zero_grad()
            student_output = ct_student(ct_input)
            with torch.no_grad():
                teacher_output = bfe_teacher(bfe_input)
                predict_loss = criterion_predict(student_output, bfe_morph_params)
                feature_difference_list = [criterion_feature_map(teacher_feat, student_feat) for (teacher_feat, student_feat) in
                                        zip(list(bfe_teacher.feature_dict.values()),
                                            list(ct_student.feature_dict.values()))]
                feature_loss = (torch.stack(feature_difference_list, dim=0) * torch.tensor(gamma_list).to(device)).sum() if keep_feature_list else 0
                ts_pred_loss = criterion_predict(teacher_output, student_output)
            if use_pred:
                val_ts_pred_loss += ts_pred_loss.item() / len(val_loader)
            val_predict_loss += predict_loss.item() / len(val_loader)
            if feature_loss > 0:
                val_feat_loss += feature_loss.item() / len(val_loader)
        if val_predict_loss < min_val_pred_loss:
            min_val_pred_loss = val_predict_loss
            save_dict = {"epoch": epoch + 1, "state_dict": ct_student.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(save_dict, f"{save_dir}/best_{epoch+1}_val_pred_loss:{min_val_pred_loss}.pth")
        if use_wandb:
            val_log_dict = {"epoch": epoch + 1, "val_loss": val_predict_loss + val_feat_loss if not use_pred else val_predict_loss + val_feat_loss + val_ts_pred_loss,"val_feature_loss": val_feat_loss,
                       "val_pred_loss": val_predict_loss}
            if use_pred:
                val_log_dict["val_ts_pred_loss"] = val_ts_pred_loss * use_pred
            wandb.log(val_log_dict)
        if (epoch + 1) % 10 == 0:
            save_dict = {"epoch": epoch + 1, "state_dict": ct_student.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(save_dict, f"{save_dir}/{epoch+1}.pth")
    return save_dir


def distill_test(exp_name, epoch_name, bfe_patch_dir, ct_patch_dir, save_dir, use_data, test_txt_path, norm_dir,
                 test_save_dir, keep_feature_list):
    bfe_morph_min_array = np.load(os.path.join(norm_dir, "morph_min.npy"), allow_pickle=True)
    bfe_morph_max_array = np.load(os.path.join(norm_dir, "morph_max.npy"), allow_pickle=True)
    batch_size = 1
    assert batch_size == 1, "test plot need batch_size=1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = DistillDataset(ct_patch_dir, bfe_patch_dir, test_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = CT_Predictor(input_channels=2, middle_channels=64, feature_channels=61, output_channels=4, type="res50", keep_feature_list=keep_feature_list).to(device)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(f"{save_dir}/{epoch_name}.pth")["state_dict"])
    model.to(device)
    model.eval()
    num_params = 4
    bfe_pts = np.zeros((len(test_dataset), num_params))
    pred_pts = np.zeros((len(test_dataset), num_params))
    test_loss = 0
    for i, ((ct_input, bfe_input), bfe_morph_params) in enumerate(test_loader):
        ct_input = ct_input.to(device)
        with torch.no_grad():
            output = model(ct_input)
            pred_pts[i, :] = output.cpu().numpy()
            bfe_pts[i, :] = bfe_morph_params
            loss = criterion(output, bfe_morph_params.to(device))
        test_loss += loss.item() / len(test_loader)
    print(f"Test loss: {test_loss}")
    if not os.path.exists(test_save_dir):
        os.mkdir(test_save_dir)
    np.save(f"{test_save_dir}/bfe_pts.npy", bfe_pts)
    np.save(f"{test_save_dir}/pred_pts.npy", pred_pts)
    for i in range(num_params):
        plt.figure()
        plt.scatter(bfe_pts[:, i], pred_pts[:, i])
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("y_true")
        plt.xlim(0, 1)
        plt.ylabel("y_pred")
        plt.ylim(0, 1)
        plt.title("r value: %.3f" % r_score(bfe_pts[:, i], pred_pts[:, i]))
        plt.savefig(f"{test_save_dir}/{i}.png")
        plt.close()

