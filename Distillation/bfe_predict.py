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


def bfe_patch_norm(bfe_patch, bfe_min=0, bfe_max=2047):
    return np.log2(bfe_patch - bfe_min + 1) / np.log2(bfe_max - bfe_min + 1 + 1e-8)


def bfe_morphology_norm(bfe_morphology, morphology_min, morphology_max):
    """
    normalize bfe morphology array
    input and return: (4,) array
    """
    return (bfe_morphology-morphology_min)/(morphology_max-morphology_min + 1e-8)


def r_score(y_pred, y_true):
    return stats.linregress(y_pred, y_true)[2]


class BFEDataset(torch.utils.data.Dataset):
    def __init__(self, bfe_patch_dir, txt_path,  morphology_min, morphology_max):
        self.bfe_patch_dir = bfe_patch_dir
        with open(txt_path, 'r') as f:
            self.txt_list = f.readlines()
        self.bfe_list = [os.path.join(self.bfe_patch_dir, txt_name.strip().split('/')[-1]) for txt_name in
                         self.txt_list]
        self.morphology_min = morphology_min
        self.morphology_max = morphology_max

    def __len__(self):
        return len(self.bfe_list)

    def __getitem__(self, idx):
        bfe_patch_path = self.bfe_list[idx]
        # print(f"bfe patch path:{bfe_patch_path}")
        bfe_patch = np.load(bfe_patch_path, allow_pickle=True)
        try:
            bfe_patch_img = bfe_patch_norm(bfe_patch["patch"])
        except Exception:
            print(f"warning!wrong file:{bfe_patch_path}")
            raise ValueError
        bfe_mask_img = bfe_patch["mask"]
        # print("bfe morph params:{}".format(bfe_patch["morph_params"]))
        bfe_morph_params = torch.tensor(bfe_morphology_norm(bfe_patch["morph_params"], self.morphology_min, self.morphology_max)).float()
        bfe_input = torch.tensor(np.stack([bfe_patch_img, bfe_mask_img], axis=0)).float()
        return bfe_input, bfe_morph_params


def output_middle_feature_hook(module, input, output, feature_dict, i):
    feature_dict[i] = output


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock3D(nn.Module):
    """
    create basic 2 res blocks as the first 2 layers of the network to downsample the input image
    and simulate the tuihua effect from high resolution to low resolution
    """

    def __init__(self, in_planes, planes, stride=2):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(in_planes, planes, stride)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.conv3(x)  # use conv to down sample the input image

        out += residual
        out = self.relu(out)

        return out


class BFE_Predictor(nn.Module):
    """
    use a pretrained 3D Resnet 50 with 2 convolutional layers before it to predict the BFE
    first 2 basic blocks make channel change from 3 to 64 and make feature map 60*60*60 to 15*15*15
    """
    def __init__(self, input_channels=2, middle_channels=64, feature_channels=61, output_channels=4, type="res50_3d",
                 keep_feature_list=None, verbose=False, bfe_pretrained_path=None):
        super(BFE_Predictor, self).__init__()
        self.feature_dict = {}
        self.keep_feature_list = keep_feature_list
        self.verbose = verbose
        self.type = type
        if type == "res50_3d":
            res_model = torch.hub.load('~/.cache/torch/hub/facebookresearch_pytorchvideo_main', 'slow_r50',source='local', pretrained=False)
            res_model.blocks[0].conv = torch.nn.Conv3d(middle_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                                           padding=(0, 3, 3))
            res_model.blocks[5].proj = torch.nn.Linear(2048, feature_channels)
            res_model.blocks[5].pool = torch.nn.AvgPool3d(kernel_size=1, stride=1, padding=0)
            if keep_feature_list is not None:
                # for i in keep_feature_list:
                #     assert isinstance(i, int)
                #     self.feature_dict[i] = None
                for i in keep_feature_list:
                    res_model.blocks[i].register_forward_hook(partial(output_middle_feature_hook, feature_dict=self.feature_dict, i=i))
            bfe_model = nn.Sequential(BasicBlock3D(input_channels, 64, stride=1),
                                      BasicBlock3D(64, middle_channels, stride=1),
                                      res_model,
                                      nn.Linear(feature_channels, output_channels),
                                      nn.Sigmoid())
            self.model = bfe_model
            if bfe_pretrained_path:
                ckpt_dict = torch.load(bfe_pretrained_path)["state_dict"]
                self.load_state_dict(ckpt_dict)
        elif type == "res50_2d":
            print(f"if use resnet 2d, please check if the last dim of data means the depth, or the number of 2d slices")
            res_model = torch.hub.load('~/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub', 'nvidia_resnet50', pretrained=True, source='local')
            res_model.conv1 = nn.Conv2d(middle_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            res_model.fc = nn.Linear(in_features=2048, out_features=61, bias=True)
            self.downsample = nn.Sequential(BasicBlock3D(input_channels, 64, stride=1),
                                      BasicBlock3D(64, middle_channels, stride=1))
            self.model = res_model
            self.avg_connect = nn.AdaptiveAvgPool1d(output_size=output_channels)
        else:
            raise NotImplementedError
        
        

    def forward(self, bfe_input):
        if bfe_input.shape[2] < 64:
            bfe_input = F.interpolate(bfe_input, size=(64, 64, 64), mode="trilinear", align_corners=True)
        if self.type == "res50_3d":
            output = self.model(bfe_input)
        elif self.type == "res50_2d":
            bfe_input = self.downsample(bfe_input)
            (N, C, H, W, D) = bfe_input.shape
            bfe_input = bfe_input.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)  
            # here the x is the num of slices, so need to the previous order is N, C, D, H, W and now should be permuted to N, D, C, H, W
            output = self.model(bfe_input)
            output = output.reshape(N, -1)
            output = self.avg_connect(output)
        else:
            raise NotImplementedError
        return output


class CT_Predictor(nn.Module):
    def __init__(self, input_channels=2, middle_channels=64, feature_channels=61, output_channels=4, type="res50", keep_feature_list=None, verbose=False):
        super(CT_Predictor, self).__init__()
        self.feature_dict = {}
        self.keep_feature_list = keep_feature_list
        if type == "res50":
            res_model = torch.hub.load('/dssg/home/acct-bmezlc/bmezlc-user4/.cache/torch/hub/facebookresearch_pytorchvideo_main', 'slow_r50',source='local', pretrained=False)
            res_model.blocks[0].conv = torch.nn.Conv3d(middle_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                                       padding=(0, 3, 3))
            res_model.blocks[5].proj = torch.nn.Linear(2048, feature_channels)
            res_model.blocks[5].pool = torch.nn.AvgPool3d(kernel_size=1, stride=1, padding=0)
            ct_model = nn.Sequential(nn.ConvTranspose3d(input_channels, middle_channels, kernel_size=2, stride=2),
                                     nn.ReLU(),
                                     nn.ConvTranspose3d(middle_channels, middle_channels, kernel_size=2, stride=2),
                                     nn.ReLU(),
                                     res_model,
                                     nn.Linear(feature_channels, output_channels),
                                     nn.Sigmoid())
            if keep_feature_list is not None:
                for i in keep_feature_list:
                    res_model.blocks[i].register_forward_hook(
                        partial(output_middle_feature_hook, feature_dict=self.feature_dict, i=i))
            self.model = ct_model
        else:
            raise NotImplementedError
        self.verbose = verbose

    def forward(self, ct_input):
        if ct_input.shape[2] < 16:
            ct_input = F.interpolate(ct_input, size=(16, 16, 16), mode="trilinear", align_corners=True)
        output = self.model(ct_input)
        return output


def bfe_single_train(exp_prefix, exp_name, train_txt_path, val_txt_path, save_dir, norm_dir, bfe_patch_dir):
    batch_size = 8
    use_wandb = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0003
    if use_wandb:
        wandb.init(project="predict_analysis", name=exp_name, settings=wandb.Settings(_disable_stats=True))
    os.makedirs(save_dir, exist_ok=True)
    bfe_morph_min_array = np.load(os.path.join(norm_dir, "morph_min.npy"), allow_pickle=True)
    bfe_morph_max_array = np.load(os.path.join(norm_dir, "morph_max.npy"), allow_pickle=True)
    train_dataset = BFEDataset(bfe_patch_dir, train_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    val_dataset = BFEDataset(bfe_patch_dir, val_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = BFE_Predictor(input_channels=2, feature_channels=61, output_channels=4, type="res50_3d", keep_feature_list=None, verbose=True).to(device)
    if use_wandb:
        wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 100], gamma=0.1)
    criterion = nn.MSELoss()
    epochs = 200
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        val_loss = 0
        for i, (bfe_input, bfe_morph_params) in enumerate(train_loader):
            bfe_input = bfe_input.to(device)
            bfe_morph_params = bfe_morph_params.to(device)
            optimizer.zero_grad()
            output = model(bfe_input)
            # print(output.shape)
            # exit()
            loss = criterion(output, bfe_morph_params)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/len(train_loader)
        print(f"train_loss: {train_loss}")
        scheduler.step()
        if use_wandb:
            wandb.log({"lr": torch.tensor(scheduler.get_last_lr()[0])})
            print(f"Epoch {epoch+1}/{epochs} train loss: {train_loss}")
            wandb.log({"epoch": epoch+1, "train_loss": train_loss})
        for i, (bfe_input, bfe_morph_params) in enumerate(val_loader):
            bfe_input = bfe_input.to(device)
            bfe_morph_params = bfe_morph_params.to(device)
            output = model(bfe_input)
            loss = criterion(output, bfe_morph_params)
            val_loss += loss.item()/len(val_loader)
        print(f"val loss: {val_loss}")
        if use_wandb:
            wandb.log({"val loss": val_loss})
        if (epoch + 1) % 10 == 0:
            save_dict = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(save_dict, f"{save_dir}/{epoch+1}.pth")
    return save_dir

def bfe_single_test(exp_name, epoch, bfe_patch_dir, save_dir, test_txt_path, norm_dir):
    test_save_dir = save_dir + "/val"
    os.makedirs(test_save_dir, exist_ok=True)
    bfe_morph_min_array = np.load(os.path.join(norm_dir, "morph_min.npy"), allow_pickle=True)
    bfe_morph_max_array = np.load(os.path.join(norm_dir, "morph_max.npy"), allow_pickle=True)
    batch_size = 1
    assert batch_size == 1, "test plot need batch_size=1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_dataset = BFEDataset(bfe_patch_dir, test_txt_path, bfe_morph_min_array, bfe_morph_max_array)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = BFE_Predictor(input_channels=2, feature_channels=61, output_channels=4, type="res50_3d", keep_feature_list=None, verbose=True).to(device)
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(f"{save_dir}/{epoch}.pth")["state_dict"])
    model.to(device)
    model.eval()
    num_params = 4
    bfe_pts = np.zeros((len(test_dataset), num_params))
    pred_pts = np.zeros((len(test_dataset), num_params))
    test_loss = 0
    for i, (bfe_input, bfe_morph_params) in enumerate(test_loader):
        bfe_input = bfe_input.to(device)
        with torch.no_grad():
            output = model(bfe_input)
            pred_pts[i, :] = output.cpu().numpy()
            bfe_pts[i, :] = bfe_morph_params
            loss = criterion(output, bfe_morph_params.to(device))
        test_loss += loss.item()/len(test_loader)
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


def main():
    # bfe_single_train()
    bfe_single_test()


if __name__ == "__main__":
    main()


























