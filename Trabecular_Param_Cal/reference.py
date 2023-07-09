""":
original reference code from Zhihao Xue
"""

import os
from posixpath import join
from pprint import pprint
import h5py
import numpy as np
import scipy
import SimpleITK as sitk
import skimage
import torch
import nibabel as nib
from scipy.ndimage import rotate
from SimpleITK.SimpleITK import TernaryMagnitudeSquared
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_local
from skimage.morphology import (remove_small_holes, remove_small_objects,
                                skeletonize)
from torchvision.transforms import InterpolationMode, Resize
from torchvision.transforms.functional import InterpolationMode as im
from torchvision.transforms.transforms import CenterCrop
from tqdm import tqdm


# from utils import show_seg, show_slices


# def save_h5(image, pars, subject):
#     assert image.shape[0] == pars.shape[0]
#     image = (image - image.min()) / (image.max() - image.min())
#     n = image.shape[0]
#     for i in range(n):
#         np.savez_compressed(join(save_path, subject + "_{}".format(i)), img=image[i], pars=pars[i])


def save_csv(subject, pars):
    n = len(pars)
    counter = 0
    counter_tbsp = 0
    tbth, tbsp = [], []
    for i in range(n):
        if pars[i][0] > 1e-6:
            tbth.append(pars[i][0])
            # tbsp.append(pars[i][1])
            counter += 1
        if pars[i][1] > 1e-6:
            tbsp.append(pars[i][1])
            counter_tbsp += 1
    # print(tbth)
    tbth = np.sum(tbth) / counter
    tbsp = np.sum(tbsp) / counter_tbsp
    bvtv = tbth / (tbth + tbsp)
    tbn = 1.0 / (tbth + tbsp)
    with open(csv_path, "a+") as f:
        f.writelines([subject, ",", str(tbth), ",", str(tbsp), ",", str(bvtv), ",", str(tbn), "\n"])


def find_img(seg):
    seg[seg != seg_map[ORG]] = 0  # 设置不感兴趣的label处的像素为0 # org英爱是类别名称，将背景的label设置为0
    n = seg.shape[0]
    slice_idx = list(range(n))
    del_idx = []
    for i in range(n):
        if torch.sum(seg[i]) < 1:  # 排除空的slice
            del_idx.append(i)
    if len(del_idx) > 0:
        for idx in del_idx:
            slice_idx.remove(idx)
    return slice_idx


def local_binary(img):
    shape = img.shape
    n = shape[0]
    _img = np.zeros(shape, dtype=int)
    for i in range(n):
        _img[i] = local_binary_pattern(image=img[i], P=1, R=5, method="ror")
    _img[_img > 0] = 1
    _img = 1 - _img
    return _img


def ada_thre(img, bs, os):  # 自适应阈值
    shape = img.shape
    n = shape[0]
    _img = np.zeros(shape, dtype=np.int32)
    for i in range(n):
        adaptive_thresh = threshold_local(img[i], block_size=bs, offset=os)
        _img[i] = img[i] > adaptive_thresh
    return _img


def remove_small_obj(img, ms):
    shape = img.shape
    n = shape[0]
    img = img.astype(np.bool8)
    _img = np.zeros(shape, dtype=np.bool8)
    for i in range(n):
        _img[i] = remove_small_objects(img[i], min_size=ms)
        _img[i] = remove_small_holes(_img[i], area_threshold=ms)
    return _img


def find_rect(img):
    n = img.shape[0]
    mask = np.zeros(img.shape, dtype=np.int32)
    for i in range(n):
        mask[i][img[i] != 0] = 1
    return mask


def cal_parameters(bi_img, seg, mask):
    """
    计算骨小梁的参数
    :param bi_img:  之前提取的骨头部分的结果
    :param seg:  分割结果
    :param mask:  注意是一个粗略的mask， 比如长方体等等，对应了bffe的有效scale范围（非背景）
    :return:
    """


    seg_idx = seg_map[ORG]  # 这个map是一个字典，将label转化为数字，对应上mask的数字
    _seg = np.zeros(seg.shape, dtype=np.int32)
    _seg[seg == seg_idx] = 1  # 拿到标注的区域

    shape = bi_img.shape
    n = shape[0]
    spac = 1 - bi_img
    spac = spac * mask  # 在大致的方形研究区域内，除了划出的骨头部分以外的区域
    # 分别分割出bone 和 space
    bone_skel = np.zeros(shape, dtype=np.int32)
    bone_dist = np.zeros(shape, dtype=np.int32)
    spac_skel = np.zeros(shape, dtype=np.int32)
    spac_dist = np.zeros(shape, dtype=np.int32)

    for i in range(n):
        # 分成每一层不同的slice进行操作
        _bone = bi_img[i]
        bone_dist[i] = scipy.ndimage.morphology.distance_transform_edt(_bone)
        # 这个计算的是体积元内任意一个点（这里是slice内任意一个点）到边界的最小距离，是欧式距离 （这个点到背景任意一点距离的最小值）
        bone_skel[i] = skeletonize(_bone).astype(np.int32)
        # 提取骨架， 目前按照文档看到的是骨架内是1，骨架以外的区域是0

        _spac = spac[i]
        spac_dist[i] = scipy.ndimage.morphology.distance_transform_edt(_spac)
        spac_skel[i] = skeletonize(_spac).astype(np.int32)

    nib.save(nib.Nifti1Image((bone_dist * _seg).astype(np.float32), np.eye(4)), 'pars\\' + str(20210126) + '_dt.nii.gz')
    bone_result = (bone_skel * bone_dist * _seg).astype(np.float32)
    # bone_result[bone_skel<1] = np.nan
    bone_result[bone_result < 0.9] = np.nan
    nib.save(nib.Nifti1Image(bone_result, np.eye(4)), 'pars\\' + str(20210126) + '_result.nii.gz')
    spac_result = (spac_skel * spac_dist * _seg).astype(np.float32)
    # spac_result[spac_skel<1] = np.nan
    spac_result[spac_result < 0.9] = np.nan

    tbth = np.nanmean(bone_result, axis=(1, 2)) * BFFE_SPACING  # nan的区域就不求均值，被忽略掉，这样得到的是每一层slice上面的均值
    tbsp = np.nanmean(spac_result, axis=(1, 2)) * BFFE_SPACING * 3  # TODO 这里为什么要乘以3

    # bvtv = tbth / (tbth + tbsp)
    # tbn = 1.0 / (tbth + tbsp)
    # np.nan_to_num(bvtv, 0)
    # np.nan_to_num(tbn, 0)
    # bvtv[bvtv > 1e3] = 0.0
    # tbn[tbn > 1e3] = 0.0
    pars = np.stack([tbth, tbsp], axis=0).transpose()
    # pars = np.stack([tbth, tbsp, bvtv, tbn], axis=0).transpose()
    return pars


def run(subject):
    # t1_file = os.path.join(mri_path, subject)
    seg_file = os.path.join(seg_path, subject[0:-7] + '2.nii.gz')
    bffe_file = os.path.join(mri_path, subject)

    # t1_data = sitk.ReadImage(t1_file)
    seg_data = sitk.ReadImage(seg_file)
    bffe_data = sitk.ReadImage(bffe_file)

    # t1_img = torch.tensor(sitk.GetArrayFromImage(t1_data), dtype=torch.int32)
    seg_img = torch.tensor(sitk.GetArrayFromImage(seg_data).astype(int), dtype=torch.int32)
    bffe_img = torch.tensor(sitk.GetArrayFromImage(bffe_data), dtype=torch.int32)

    slice_idx = find_img(seg_img)  # 找到骨小梁所在的slice index
    # print(slice_idx)

    space = bffe_data.GetSpacing()[0] # 0.2188
    scale_bffe = space / BFFE_SPACING # 0.2188/0.1
    # scale_t1 = space / T1_SPACING
    size_bffe = (bffe_img.shape[0], int(bffe_img.shape[1] * scale_bffe), int(bffe_img.shape[2] * scale_bffe))
    # 0.2188/0.1 *640, around 1300

    # size_t1 = (t1_img.shape[0], int(t1_img.shape[1] * scale_t1), int(t1_img.shape[2] * scale_t1))
    # 选择出来研究的slice做resize和裁剪
    # t1_img = CenterCrop([512, 512])(Resize(size=size_t1[1:], interpolation=im.BILINEAR)(t1_img)).numpy()[slice_idx]
    # bffe_bk = CenterCrop([512, 512])(Resize(size=size_t1[1:], interpolation=im.BILINEAR)(bffe_img)).numpy()[slice_idx]
    seg_img = Resize(size=size_bffe[1:], interpolation=im.NEAREST)(seg_img).numpy()[slice_idx]
    bffe_img = Resize(size=size_bffe[1:], interpolation=im.BILINEAR)(bffe_img).numpy()[slice_idx]
    # resize to around 1300

    mask_bffe = find_rect(bffe_img)  # 得到ROI的mask
    # mask_t1 = find_rect(bffe_bk)

    # t1_img = t1_img * mask_t1
    binary_img = ada_thre(bffe_img, bs=11, os=1) * mask_bffe  # 研究threshold下mask和bffe区域内的mask的交集，有效骨小梁区域
    binary_img = remove_small_obj(binary_img, ms=4)

    # show_slices(binary_img, "img/bffe_b.png")
    # show_slices(bffe_img, "img/bffe.png")
    # exit()

    pars = cal_parameters(binary_img, seg_img, mask_bffe)  # 计算骨小梁参数
    # print(pars)
    # show_slices(t1_img, "img/t1.png")
    # print(t1_img.shape)
    # print(pars)
    # exit()
    save_csv(subject, pars)
    # save_h5(t1_img, pars, subject)

    # show_seg(torch.tensor(bffe_img), torch.tensor(seg_img), "bffe.png")
    # show_slices(binary_img, "bffe_b.png")
    # nib.save(nib.Nifti1Image(t1_img, np.eye(4)), 'pars\\' + str(subject) + '_orig.nii.gz')
    # affine_array = np.array([[1.11316071e-02, -5.30286951e-03, 7.48809692e-01,
    #                           -1.32700165e+02],
    #                          [-2.18465168e-01, 5.18203673e-04, 3.82197553e-02,
    #                           1.23873642e+02],
    #                          [-7.87611553e-04, -2.18685101e-01, -1.80672320e-02,
    #                           5.49174690e+01],
    #                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #                           1.00000000e+00]])
    nib.save(nib.Nifti1Image((binary_img * seg_img).astype('uint8'), np.eye(4)),
             'pars\\' + str(subject) + '_bin.nii.gz')  # 保存resize后的骨小梁分割
    nib.save(nib.Nifti1Image(bffe_img, np.eye(4)), 'pars\\' + str(subject) + '.nii.gz')  # 保存resize后的bffe image
    nib.save(nib.Nifti1Image((binary_img * seg_img * bffe_img), np.eye(4)),
             'pars\\' + str(subject) + '_trabecular.nii.gz')
    # show_slices(t1_img, "t1.png")
    # show_slices(mask, "mask.png")
    # pprint(pars)


if __name__ == "__main__":
    BFFE_SPACING = 0.1
    T1_SPACING = 0.3125
    mri_path = "test"  # 影像所在目录
    seg_path = "label revised 2"  # ROI分割所在目录 # TODO1: 分割结果不知道需不需要自己训练得到
    seg_map = {"femur1_": 1}
    subjects = os.listdir(mri_path)  # 可根据需要修改
    ORG = "femur1_"  # 可根据需要修改
    # ORG = "tibia2"
    save_path = "pars/{}/".format(ORG)
    csv_path = "{}.csv".format(ORG)
    os.system(f"rm -rf {csv_path}*")
    os.system(f"rm -rf {save_path}*")

    for s in tqdm(subjects, position=1, leave=False):
        run(s)

    # for s in subjects:
    #     run(s)
