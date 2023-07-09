from operator import index
import os
from tkinter.tix import InputOnly
import SimpleITK as sitk
import ants
import numpy as np
import cv2 as cv

import logging
from tqdm import tqdm

import shutil

method = 'DenseRigid'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(f"log_with_mask_{method}.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
 
logger.addHandler(handler)
logger.addHandler(console)
 
logger.info("Start print log")


ct_path = 'samples/data/CT'
bfe_path = 'samples/data/reg-BFE'  # registered BFE nii images
fake_ct_path = 'samples/data/fake_CT_masked'
ct_tmp_path = f'ct_tmp_{method}_cycle_ada'
fake_ct_tmp_path = f'fake_ct_tmp_{method}_cycle_ada'
ct_mask_path = 'samples/data/mask_seg'
fake_ct_mask_path = 'samples/data/fake_mask_seg'
save_syn_mask_path = 'samples/data/syn_mask'

os.makedirs(save_syn_mask_path, exist_ok=True)
if not os.path.isdir(ct_tmp_path):
    os.makedirs(ct_tmp_path)
if not os.path.isdir(fake_ct_tmp_path):
    os.makedirs(fake_ct_tmp_path)
save_path = f'results_{method}_cycleada'
# save_path_debug = f'results_{method}_cycleada_debug'


# on bones, the threshold is 100 to get the boundary

# bone_thres = 250
# bone_thres_fake = 300
bg_value = -200 # background

def binary(img, thres):
    mask = np.ones(img.shape)
    mask[img<=thres] = 0
    return mask.astype(np.int16)

def copy_itk_img_info(new, old, input_array=False):
    '''
    old mast be an image
    if new is an array,input_array should be set to True
    if an image, False
    '''
    if input_array:
        new = sitk.GetImageFromArray(new)
    new.SetDirection(old.GetDirection())
    new.SetSpacing(old.GetSpacing())
    new.SetOrigin(old.GetOrigin())
    return new

def copy_ants_img_info(new, old, input_array=False):
    if input_array:
        new = ants.from_numpy(new, origin=old.origin, spacing=old.spacing, direction=old.direction,\
             has_components=False, is_rgb=False)
        return new
    new.set_direction(old.direction)
    new.set_origin(old.origin)
    new.set_spacing(old.spacing)
    return new

def mean_coord_3d(mask_array):
    # mask: (z, x, y) as shape:(N, 512, 512)
    assert mask_array.ndim == 3
    return tuple(map(lambda x:x.mean(), mask_array.nonzero()))
    # very fast implemention, be careful that nonzero is different in pytorch and numpy

def get_topk_biggest_component(mask_old, topk=3):
    """
    mask_old: an sitk image of which topk connected components are needed
    return: al list [], in which element is (mask, size)
    mask is an image array (not sitk image) and size is its physical size
    """
    logger.info(f"start getting the biggest {topk} components!")
    cc = sitk.ConnectedComponent(mask_old)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, mask_old)
    topk_mask_list = []
    for label in stats.GetLabels():
        size = stats.GetPhysicalSize(label)
        labelmaskimage = sitk.GetArrayFromImage(cc)
        component = (labelmaskimage==label).astype(np.int16)
        # component = sitk.GetImageFromArray((labelmaskimage==label).astype(np.int16))
        # component = copy_itk_img_info(component, mask_old)
        # use array to return in this version for easier usage
        topk_mask_list.append((component, size))
    topk_mask_list = sorted(topk_mask_list, key = lambda x:x[1],reverse=True)
    if len(topk_mask_list) > topk:
        topk_mask_list = topk_mask_list[:topk]
    logger.info(f"finally got {len(topk_mask_list)} components for topk:{topk} task!")
    for i, (mask,size) in enumerate(topk_mask_list):
        logger.info(f"top{topk}_component No.{i}: size {size}, center:{mean_coord_3d(mask)} listed!")
    return topk_mask_list # (from biggest to smallest) [(mask,size)]

def apply_mask(ct_array, mask_needed, bg_value=-200):
    '''
    ct_array: array
    mask_needed: array
    bg_thres: all negative position has the value of init_thres
    '''
    img_needed = ct_array.copy()
    img_needed[mask_needed==0] = bg_value
    return img_needed


def open_process(mask_old, size=2):
    filter = sitk.BinaryMorphologicalOpeningImageFilter()
    filter.SetKernelRadius(size)
    mask_new = filter.Execute(mask_old)
    return mask_new


# def segment(mask_total, mask_save_path, circle_surrounding=True, retry=True, use_y_division=False):
#     '''
#     mask_total: an sitk image
#     return : array
#     '''
#     if not os.path.isdir(mask_save_path):
#         os.makedirs(mask_save_path)
#     logger.info("start segmentation for the leg!")
#     top2_mask_list = get_topk_biggest_component(mask_total, topk=2)
#     if len(top2_mask_list) != 2:
#         logger.warning(f"in segment function, only {len(top2_mask_list)} components got!")
#         return -1,- 1
#     else:
#         # the upper component has a a bigger mean z coordinate
#         if use_y_division:
#             upper_then_lower_list = sorted(top2_mask_list, key=lambda x: mean_coord_3d(x[0])[1], reverse=False) # for BFE direction, the upper part has a smaller mean y coordinate
#         else:
#             upper_then_lower_list = sorted(top2_mask_list, key = lambda x:mean_coord_3d(x[0])[0],reverse=True)
#         mask_upper = upper_then_lower_list[0][0]
#         mask_upper_size = upper_then_lower_list[0][1]
#         mask_lower = upper_then_lower_list[1][0]
#         mask_lower_size = upper_then_lower_list[1][1]
#
#         for mask_component_size in [mask_upper_size, mask_lower_size]:
#             if mask_component_size < 5e3 or mask_component_size > 1e5:
#                 logger.warning(f"in segment function, the component size is {mask_component_size}!")
#                 mask_total = open_process(copy_itk_img_info(sorted(top2_mask_list, key = lambda x:x[1],reverse=True)[0][0], mask_total, input_array=True), size=6)
#                 if retry:
#                     logger.info("retry segmentation!")
#                     return segment(mask_total, mask_save_path, circle_surrounding, retry=False, use_y_division=use_y_division)
#                 else:
#                     logger.info("no more retry!")
#         sitk.WriteImage(sitk.GetImageFromArray(mask_upper), os.path.join(mask_save_path, 'mask_upper.nii.gz'))
#         sitk.WriteImage(sitk.GetImageFromArray(mask_lower), os.path.join(mask_save_path, 'mask_lower.nii.gz'))
#         if use_y_division:
#             mask_upper = mask_upper.transpose(1,0,2)
#             mask_lower = mask_lower.transpose(1,0,2)
#             sitk.WriteImage(sitk.GetImageFromArray(mask_upper), os.path.join(mask_save_path, 'mask_upper_transposed.nii.gz'))
#             sitk.WriteImage(sitk.GetImageFromArray(mask_lower), os.path.join(mask_save_path, 'mask_lower_transposed.nii.gz'))
#             logger.info("use y division applied!")
#         logger.info(f"mask upper mean cordinate: {mean_coord_3d(mask_upper)}")
#         logger.info(f"mask lower mean cordinate: {mean_coord_3d(mask_lower)}")
#         (z_shape, y_shape, x_shape) = mask_upper.shape
#         if use_y_division:
#             upper_max_z = np.any(mask_upper, axis=(1, 2)).nonzero()[0].max()
#             logger.info(f"choosing the upper_max_z: {upper_max_z}")
#         else:
#             upper_min_z = np.any(mask_upper, axis=(1, 2)).nonzero()[0].min()
#             logger.info(f"choosing the upper_min_z: {upper_min_z}")
#         # find the min z on the upper bone(z is the index, not the itk image z, and starts from 0)
#         # lower_max_z = np.argmax(np.nonzero(np.any(mask_lower, axis=0).astype(np.int16))[0])
#         # find the max z on the lower bone
#         if use_y_division:
#             mask_upper_part = np.tile(np.concatenate((np.ones(upper_max_z+1), np.zeros(z_shape-upper_max_z-1))),(x_shape, y_shape, 1)).transpose((2,1,0))
#         else:
#             mask_upper_part = np.tile(np.concatenate((np.zeros(z_shape-upper_min_z-1), np.ones(upper_min_z+1))),(x_shape, y_shape, 1)).transpose((2,1,0))
#         # sitk.WriteImage(sitk.GetImageFromArray(mask_upper_part), os.path.join(mask_save_path, 'mask_upper_part.nii.gz'))
#         logger.info(f"mask upper part shape:{mask_upper_part.shape}")
#         if not circle_surrounding:
#             mask_top = ((mask_upper_part - mask_lower) > 0).astype(np.int16)
#         else:
#             cycle_ada_outline = get_cycle_ada_outline(mask_lower, input_array=True)
#             logger.info(f"cycle ada outline max_z: {np.any(cycle_ada_outline, axis=(1, 2)).nonzero()[0].max()}")
#             mask_top = ((mask_upper_part-cycle_ada_outline)>0).astype(np.int16)
#             # get the convex and fully-connected outline component of (the intersection of upperpart&&lowerbone)
#         mask_bottom = np.ones((z_shape, y_shape, x_shape)) - mask_top
#         if not use_y_division:
#             logger.info("mask bottom max z:{}".format(np.any(mask_bottom, axis=(1, 2)).nonzero()[0].max()))
#         else:
#             logger.info("mask bottom min z:{}".format(np.any(mask_bottom, axis=(1, 2)).nonzero()[0].min()))
#         if use_y_division:
#             mask_top = mask_top.transpose(1, 0, 2)
#             mask_bottom = mask_bottom.transpose(1, 0, 2)
#         sitk.WriteImage(sitk.GetImageFromArray(mask_top), os.path.join(mask_save_path, 'mask_top.nii.gz'))
#         logger.info(f"mask_top.shape:{mask_top.shape}")
#         sitk.WriteImage(sitk.GetImageFromArray(mask_bottom), os.path.join(mask_save_path, 'mask_bottom.nii.gz'))
#         # print(os.path.join(mask_save_path, 'mask_bottom.nii.gz'), "is saved!") # debug only
#     return mask_top, mask_bottom


def registration_and_apply(fixed_image_path, moving_image_path, to_apply_image_path, mask_img_path, method, save_dir, save_all=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fix_img = ants.image_read(fixed_image_path)
    logger.info(f"fixed image shape:{fix_img.shape}")
    move_img = ants.image_read(moving_image_path)
    logger.info(f"moving image shape:{move_img.shape}")
    to_apply_img = ants.image_read(to_apply_image_path)
    logger.info(f"to apply image shape:{to_apply_img.shape}")
    mask_img = ants.image_read(mask_img_path)
    logger.info(f"mask image shape:{mask_img.shape}")

    outs = ants.registration(fix_img, move_img, type_of_transform=method, mask=mask_img)

    logger.info(f"{method} registration done between ct:{fixed_image_path} and fake ct:{moving_image_path}!")

    fake_img = outs['warpedmovout']

    fake_img.set_direction(fix_img.direction)
    fake_img.set_origin(fix_img.origin)
    fake_img.set_spacing(fix_img.spacing)
    if save_all:
        ants.image_write(fake_img, os.path.join(save_dir, 'fake_ct_registered.nii.gz'))

    # apply the transformation to the to_apply_image
    move2fix_matrix_list = outs['fwdtransforms']
    applied_img = ants.apply_transforms(fixed=fix_img, moving=to_apply_img, transformlist=move2fix_matrix_list,
                                       interpolator="linear")

    applied_img.set_direction(fix_img.direction)
    applied_img.set_origin(fix_img.origin)
    applied_img.set_spacing(fix_img.spacing)

    ants.image_write(applied_img, os.path.join(save_dir, 'applied_bfe.nii.gz'))

    logger.info(f"{method} registration done between to_apply_bfe and ct!")
    if save_all:
        shutil.copy(move2fix_matrix_list[0], os.path.join(save_dir, 'fwdtransforms.mat'))
    # ants.image_write(move2fix_matrix, os.path.join(save_dir, 'fwdtransforms.mat'))

    fix2move_matrix_list = outs['invtransforms']
    # print(fix2move_matrix)
    if save_all:
        shutil.copy(fix2move_matrix_list[0], os.path.join(save_dir, 'invtransforms.mat'))
    # ants.image_write(fix2move_matrix, os.path.join(save_dir, 'invtransforms.mat'))
    logger.info(f"finish ANTs registration for image: {fixed_image_path}!")
    return applied_img


def get_mask_for_synthesis(mask_top_img, mask_bottom_img):
    mask_top_array = sitk.GetArrayFromImage(mask_top_img)
    logger.info(f"mask_top_array.shape:{mask_top_array.shape}")
    (z_shape, y_shape, x_shape) = mask_top_array.shape
    mask_bottom_array = sitk.GetArrayFromImage(mask_bottom_img)
    upper_min_z = np.any(mask_top_array, axis=(1, 2)).nonzero()[0].min()
    mask_upper_part = np.tile(np.concatenate((np.zeros(upper_min_z + 1), np.ones(z_shape - upper_min_z - 1))),
                              (x_shape, y_shape, 1)).transpose((2, 1, 0))
    logger.info(f"mask_upper_part.shape:{mask_upper_part.shape}")
    mask_top_new_array = ((mask_upper_part - mask_bottom_array) > 0).astype(np.int16)
    mask_top_new = copy_itk_img_info(mask_top_new_array, mask_top_img, input_array=True)
    mask_bottom_array = np.ones((z_shape, y_shape, x_shape)) - mask_top_new_array
    mask_bottom_new = copy_itk_img_info(mask_bottom_array, mask_bottom_img, input_array=True)
    return mask_top_new, mask_bottom_new


def synthesis(bfe_top, bfe_bottom, mask_top, save_path):
    bfe_array_top = bfe_top.numpy(single_components=False)
    logger.info(f"bfe image array (top) shape:{bfe_array_top.shape}")
    bfe_array_bottom = bfe_bottom.numpy(single_components=False)
    logger.info(f"bfe image array (bottom) shape:{bfe_array_bottom.shape}")
    if bfe_array_top.shape != bfe_array_bottom.shape:
        logger.warning("shape not consistent between bfe_top and bfe_bottom!")
        logger.warning(f"top shape:{bfe_array_top.shape} and bottom shape:{bfe_array_bottom.shape}")
        return
    # if (mask_top + mask_bottom).astype(np.int16) != np.ones(bfe_array_top.shape).astype(np.int16):
    #     logger.warning("the sum of mask_top and mask_bottom is not 1 everywhere!")
    #     return
    bfe_total = bfe_array_top.copy()
    # print(bfe_total.shape)
    # mask_top = mask_top.transpose((2, 1, 0))
    # print(mask_top.shape)
    # ants.image_write(copy_ants_img_info(mask_top, bfe_top, input_array=True), './mask_top.nii.gz')
    mask_top = sitk.GetArrayFromImage(mask_top).astype(np.int16).transpose((2, 1, 0))  # different sequence of dims in ants_img ands ikt_img
    bfe_total[mask_top < 1] = bfe_array_bottom[mask_top < 1]
    bfe_total = copy_ants_img_info(bfe_total, bfe_top, input_array=True)
    ants.image_write(bfe_total, save_path)


def padding(img, pad_shape = (150, 300, 300), bg_value=-200):
    """
    img is sitk image
    :param img:
    :param pad_shape:
    :return:
    """
    save_array = (np.ones(pad_shape) * bg_value).astype(np.float32)
    img_array = sitk.GetArrayFromImage(img)
    img_shape = img_array.shape
    logger.info(f"in padding, original img_array.shape:{img_array.shape}")
    for i in range(img_shape[2]):
        new_i = i + (pad_shape[2]-img_shape[2]) // 2
        to_pad_shape = (((pad_shape[0] - img_shape[0])//2,)*2, ((pad_shape[1] - img_shape[1])//2,)*2)
        save_array[:, :, new_i] = np.pad(img_array[:, :, i], to_pad_shape, constant_values=bg_value)
    logger.info(f"in padding, padded img_array.shape:{save_array.shape}")
    save_img = copy_itk_img_info(save_array, img, input_array=True)
    return save_img


def main():
    assert sorted(os.listdir(ct_path)) == sorted(list(map(lambda x: x.replace(".gz", ""), os.listdir(bfe_path))))
    file_list = os.listdir(ct_path)
    for file in tqdm(file_list):
        logger.info(f"start processing file:{file}")
        try:
            index = int(file.split('.')[0])
        except Exception:
            index = file.split('.')[0]
            logger.warning(f"wrong file name!{file} with index {index} and stop processing this image!")
            continue
        file_name = file.replace('.nii', '')
        ct = sitk.ReadImage(os.path.join(ct_path, file))
        ct_array = sitk.GetArrayFromImage(ct)
        logger.info(f"ct image array shape:{ct_array.shape}")
        bfe_img_path = os.path.join(bfe_path, file+".gz")
        bfe = sitk.ReadImage(bfe_img_path)
        bfe_array = sitk.GetArrayFromImage(bfe)
        logger.info(f"bfe image array shape:{bfe_array.shape}")
        mask_top_ori_img_path = os.path.join(ct_mask_path, f"{file_name}_top.nii")
        mask_top_img = padding(sitk.ReadImage(mask_top_ori_img_path), bg_value=0)
        mask_top_img_path = os.path.join(ct_mask_path, f"{file_name}_top_pad.nii")
        sitk.WriteImage(mask_top_img, mask_top_img_path)
        mask_bottom_ori_img_path = os.path.join(ct_mask_path, f"{file_name}_bottom.nii")
        mask_bottom_img = padding(sitk.ReadImage(mask_bottom_ori_img_path), bg_value=0)
        mask_bottom_img_path = os.path.join(ct_mask_path, f"{file_name}_bottom_pad.nii")
        sitk.WriteImage(mask_bottom_img, mask_bottom_img_path)
        logger.info(f"start processing ct image with shape:{ct_array.shape}")

        fake_ct_top_img_path = os.path.join(fake_ct_path, file_name + '_top.nii')
        fake_ct_bottom_img_path = os.path.join(fake_ct_path, file_name + '_bottom.nii')

        img_top = copy_itk_img_info(apply_mask(ct_array, mask_top_img, bg_value=bg_value), ct, input_array=True)
        logger.info(f"img_top shape:{sitk.GetArrayFromImage(img_top).shape}")
        img_top_save_path = os.path.join(ct_tmp_path, f'{index}_top.nii')
        sitk.WriteImage(img_top, img_top_save_path)
        logger.info(f"top image(index:{index}) saved!")
        img_bottom = copy_itk_img_info(apply_mask(ct_array, mask_bottom_img, bg_value=bg_value), ct, input_array=True)
        img_bottom_save_path = os.path.join(ct_tmp_path, f'{index}_bottom.nii')
        sitk.WriteImage(img_bottom, img_bottom_save_path)
        logger.info(f"bottom image(index:{index}) saved!")

        mask_syn_top, mask_syn_bottom = get_mask_for_synthesis(mask_top_img, mask_bottom_img)
        mask_syn_top_save_path = os.path.join(save_syn_mask_path, f'{index}_top.nii')
        mask_syn_bottom_save_path = os.path.join(save_syn_mask_path, f'{index}_bottom.nii')
        sitk.WriteImage(mask_syn_top, mask_syn_top_save_path)
        sitk.WriteImage(mask_syn_bottom, mask_syn_bottom_save_path)

        # logger.info(f"fake top image(index:{index}) saved!")
        # img_fake_bottom = copy_itk_img_info(apply_mask(fake_ct_array, mask_fake_bottom, bg_value=bg_value), fake_ct, input_array=True)
        # img_fake_bottom_save_path = os.path.join(fake_ct_tmp_path, f'{index}_bottom.nii')
        # sitk.WriteImage(img_fake_bottom, img_fake_bottom_save_path)
        # logger.info(f"fake bottom image(index:{index}) saved!")

        logger.info("start registration between: ct_top(fix) and bfe(move)!")
        bfe_top = registration_and_apply(img_top_save_path, fake_ct_top_img_path, bfe_img_path, mask_top_img_path, method, os.path.join(save_path, f'{index}/top'), save_all=True)
        logger.info("finish registration between: ct_top(fix) and bfe(move)!")
        logger.info("start registration between: ct_bottom(fix) and bfe(move)!")
        bfe_bottom = registration_and_apply(img_bottom_save_path, fake_ct_bottom_img_path, bfe_img_path, mask_bottom_img_path, method, os.path.join(save_path, f'{index}/bottom'), save_all=True)
        logger.info("finish registration between: ct_bottom(fix) and bfe(move)!")
        logger.info("start synthesis of bfe_top and bfe_bottom")
        if not os.path.isdir(os.path.join(save_path, 'total')):
            os.makedirs(os.path.join(save_path, 'total'))
        # exit()
        synthesis(bfe_top, bfe_bottom, mask_syn_top, os.path.join(save_path, f'total/{index}.nii'))
        logger.info(f"finish synthesis of bfe_top and bfe_bottom")
        logger.info(f"finish registration for {index}.nii")
            # break # debug only
        # except Exception as e:
        #     logger.error(f"error occurred in registration for {file}!!!\n{e}")
        #     continue
        # break # debug only


if __name__ == '__main__':
    main()


