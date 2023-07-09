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

method = 'Rigid'
# method = 'SyNRA'
# method = 'SyN'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(f"log_cycle_ada_{method}.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
 
logger.addHandler(handler)
logger.addHandler(console)
 
logger.info("Start print log")


ct_path = 'samples/data/CT'
bfe_path = 'samples/data/fake_CT_30'
ct_tmp_path = f'ct_tmp_{method}_cycle_ada'
mask_save_dir = f'mask_{method}_cycle_ada'
if not os.path.isdir(ct_tmp_path):
    os.makedirs(ct_tmp_path)
if not os.path.isdir(mask_save_dir):
    os.makedirs(mask_save_dir)
save_path = f'results_{method}_cycleada'
# save_path_debug = f'results_{method}_cycleada_debug'


# on bones, the threshold is 100 to get the boundary

bone_thres = 200
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

def get_cyclinder_outline(mask_init, input_array=True):
    '''
    function: get cyclinder outline for each slice, the cyclinder can surround all points along z axis
    implement method: get the (x,y) coordinates by np.any(), then use cv2 get the min circle
    after that the min volumn of cyclinder(圆柱体) can be found easily

    assumption: (z,y,x) image
    inpu_array=True->input is an array, else is an image
    output: needed mask, is an np array
    '''
    # failed attempt to get the min convex hull

    # if input_array:
    #     mask_init = sitk.GetImageFromArray(mask_init)
    # h_convex_filter = sitk.HConvexImageFilter()
    # h_convex_filter.SetFullyConnected(False)
    # h_convex_filter.SetHeight(1e-3)
    # mask_convex = h_convex_filter.Execute(mask_init)
    # sitk.WriteImage(mask_init, './mask_init.nii.gz') # debug only

    # get the circle outline for each slice
    if not input_array:
        mask_init = sitk.GetArrayFromImage(mask_init)
    (z_shape, y_shape, x_shape) = mask_init.shape
    z_arange = np.tile(np.any(mask_init, axis=(1,2)).astype(np.int16)[:,np.newaxis,np.newaxis], (1, y_shape, x_shape))  # get the z range that exists mask_init[x0, y0, z] > 0
    yx_slice = np.any(mask_init, axis=0).astype(np.int16) # into a 2d slice
    logger.info(f'yx_slice shape:{yx_slice.shape}')
    y_array, x_array = yx_slice.nonzero()
    points_set = np.concatenate((y_array[:,np.newaxis], x_array[:,np.newaxis]), axis=1)[:, np.newaxis, :]
    (y_center,x_center),radius = cv.minEnclosingCircle(points_set) # be careful that this is a (y,x) image
    y = np.arange(y_shape)
    x = np.arange(x_shape)
    Y, X = np.meshgrid(y,x)
    needed_slice = (((Y-y_center)**2 + (X-x_center)**2) < radius**2).astype(np.int16)
    mask_cyclinder = np.tile(needed_slice, (z_shape, 1, 1)) * z_arange
    # sitk.WriteImage(sitk.GetImageFromArray(mask_cyclinder), './mask_cyclinder.nii.gz') # debug_only
    return mask_cyclinder

def get_cycle_ada_outline(mask_init, input_array=True):
    '''
    function: get cycle outline for each slice, the cycle can surround all points only in this z slice
    implement method: for each slice, use cv2 method to find the smallest cycle

    assumption: (z,y,x) image
    inpu_array=True->input is an array, else is an image
    output: needed mask, is an np array
    '''
    if not input_array:
        mask_init = sitk.GetArrayFromImage(mask_init)
    (z_shape, y_shape, x_shape) = mask_init.shape
    mask_cycle_ada = np.zeros(mask_init.shape)
    for current_z in range(z_shape):
        yx_slice = mask_init[current_z,:,:] # into a 2d slice
        # logger.info(f'yx_slice shape:{yx_slice.shape}')
        y_array, x_array = yx_slice.nonzero()
        points_set = np.concatenate((y_array[:,np.newaxis], x_array[:,np.newaxis]), axis=1)[:, np.newaxis, :]
        (y_center,x_center),radius = cv.minEnclosingCircle(points_set) # be careful that this is a (y,x) image
        y = np.arange(y_shape)
        x = np.arange(x_shape)
        Y, X = np.meshgrid(y,x)
        needed_slice = (((Y-y_center)**2 + (X-x_center)**2) < radius**2).astype(np.int16)
        mask_cycle_ada[current_z,:,:] = needed_slice
    # sitk.WriteImage(sitk.GetImageFromArray(mask_cycle_ada), './mask_cycle_ada.nii.gz') # debug_only
    return mask_cycle_ada



def segment(mask_total, mask_save_path, circle_surrounding=True):
    '''
    mask_total: an sitk image 
    return : array
    '''
    if not os.path.isdir(mask_save_path):
        os.makedirs(mask_save_path)
    (z_shape, y_shape, x_shape) = sitk.GetArrayFromImage(mask_total).shape
    logger.info("start segmentation for the leg!")
    top2_mask_list = get_topk_biggest_component(mask_total, topk=2)
    if len(top2_mask_list) != 2:
        logger.warning(f"in segment function, only {len(top2_mask_list)} components got!")
        return -1,-1
    else:
        # the upper component has a a bigger mean z coordinate 
        upper_then_lower_list = sorted(top2_mask_list, key = lambda x:mean_coord_3d(x[0])[0],reverse=True)
        mask_upper = upper_then_lower_list[0][0]
        sitk.WriteImage(sitk.GetImageFromArray(mask_upper), os.path.join(mask_save_path, 'mask_upper.nii.gz'))
        mask_lower = upper_then_lower_list[1][0]
        sitk.WriteImage(sitk.GetImageFromArray(mask_lower), os.path.join(mask_save_path, 'mask_lower.nii.gz'))
        upper_min_z = np.min(np.nonzero(np.any(mask_upper, axis=0).astype(np.int16))[0])
        logger.info(f"choosing the upper_min_z: {upper_min_z}")
        # find the min z on the upper bone(z is the index, not the itk image z, and starts from 0)
        # lower_max_z = np.argmax(np.nonzero(np.any(mask_lower, axis=0).astype(np.int16))[0])
        # find the max z on the lower bone
        mask_upper_part = np.tile(np.concatenate((np.zeros(z_shape-upper_min_z-1), np.ones(upper_min_z+1))),(x_shape, y_shape, 1)).transpose((2,1,0))
        sitk.WriteImage(sitk.GetImageFromArray(mask_upper_part), os.path.join(mask_save_path, 'mask_upper_part.nii.gz'))
        logger.info(f"mask shape:{mask_upper_part.shape}")
        if not circle_surrounding:
            mask_top = ((mask_upper_part - mask_lower) > 0).astype(np.int16)
        else:
            mask_top = ((mask_upper_part-get_cycle_ada_outline(mask_lower, input_array=True))>0).astype(np.int16)
            # get the convex and fully-connected outline component of (the intersection of upperpart&&lowerbone)
        sitk.WriteImage(sitk.GetImageFromArray(mask_top), os.path.join(mask_save_path, 'mask_top.nii.gz'))
        logger.info(f"mask_top.shape:{mask_top.shape}")
        mask_bottom = np.ones((z_shape, y_shape, x_shape))-mask_top
        sitk.WriteImage(sitk.GetImageFromArray(mask_bottom), os.path.join(mask_save_path, 'mask_bottom.nii.gz'))
    return mask_top, mask_bottom

def registration(fixed_image_path, moving_image_path, method, save_dir, save_all=True):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fix_img = ants.image_read(fixed_image_path)
    move_img = ants.image_read(moving_image_path)
    outs = ants.registration(fix_img, move_img, type_of_transform=method)

    reg_img = outs['warpedmovout']

    reg_img.set_direction(fix_img.direction)
    reg_img.set_origin(fix_img.origin)
    reg_img.set_spacing(fix_img.spacing)
    if save_all:
        ants.image_write(reg_img, os.path.join(save_dir, 'warpedmovout.nii.gz'))

    move2fix_matrix = outs['fwdtransforms'][0]
    # print(move2fix_matrix)
    if save_all:
        shutil.copy(move2fix_matrix, os.path.join(save_dir, 'fwdtransforms.nii.gz'))
    # ants.image_write(move2fix_matrix, os.path.join(save_dir, 'fwdtransforms.mat'))

    fix2move_matrix = outs['invtransforms'][0]
    # print(fix2move_matrix)
    if save_all:
        shutil.copy(fix2move_matrix, os.path.join(save_dir, 'invtransforms.nii.gz'))
    # ants.image_write(fix2move_matrix, os.path.join(save_dir, 'invtransforms.mat'))
    logger.info(f"finish ANTs registration for image: {fixed_image_path}!")
    return reg_img


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
    mask_top = mask_top.transpose((2,1,0)) # different sequence of dims in ants_img ands ikt_img
    # print(mask_top.shape)
    # ants.image_write(copy_ants_img_info(mask_top, bfe_top, input_array=True), './mask_top.nii.gz')
    bfe_total[mask_top < 1] = bfe_array_bottom[mask_top < 1]
    bfe_total = copy_ants_img_info(bfe_total, bfe_top, input_array=True)
    ants.image_write(bfe_total, save_path)



def main():
    assert os.listdir(ct_path) == os.listdir(bfe_path)
    file_list = os.listdir(ct_path)
    error_file_dict = {}
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
        logger.info(f"start processing ct image with shape:{ct_array.shape}")
        # (N, 512, 512)
        mask_total = copy_itk_img_info(binary(ct_array, bone_thres), ct, input_array=True) # get the boundary of the bones
        mask_total = open_process(mask_total,size=1)
        mask_top, mask_bottom = segment(mask_total, os.path.join(mask_save_dir,str(index)))
        if not isinstance(mask_top, np.ndarray):
            logger.warning(f"error occurred in segmentation of two bones for: \n {file}!!!")
            error_file_dict[file] = 'error:0'
            # break #debug only
            continue
        img_top = copy_itk_img_info(apply_mask(ct_array, mask_top, bg_value=bg_value), ct, input_array=True)
        img_top_save_path = os.path.join(ct_tmp_path, f'{index}_top.nii')
        sitk.WriteImage(img_top, img_top_save_path)
        logger.info(f"top image(idnex:{index}) saved!")
        img_bottom = copy_itk_img_info(apply_mask(ct_array, mask_bottom, bg_value=bg_value), ct, input_array=True)
        img_bottom_save_path = os.path.join(ct_tmp_path, f'{index}_bottom.nii')
        sitk.WriteImage(img_bottom, img_bottom_save_path)
        logger.info(f"bottom image(idnex:{index}) saved!")
        logger.info("start registration between: ct_top(fix) and bfe(move)!")
        bfe_file_path = os.path.join(bfe_path, file)
        bfe_top = registration(img_top_save_path, bfe_file_path, method, os.path.join(save_path, f'{index}/top'), save_all=True)
        logger.info("finish registration between: ct_top(fix) and bfe(move)!")
        logger.info("start registration between: ct_bottom(fix) and bfe(move)!")
        bfe_bottom = registration(img_bottom_save_path, bfe_file_path, method, os.path.join(save_path, f'{index}/bottom'), save_all=True)
        logger.info("finish registration between: ct_bottom(fix) and bfe(move)!")
        logger.info("start synthesis of bfe_top and bfe_bottom")
        if not os.path.isdir(os.path.join(save_path, 'total')):
            os.makedirs(os.path.join(save_path, 'total'))
        synthesis(bfe_top, bfe_bottom, mask_top, os.path.join(save_path, f'total/{index}.nii'))
        logger.info(f"finish synthesis of bfe_top and bfe_bottom")
        logger.info(f"finish registration for {index}.nii")
        break # debug only
    logger.info(error_file_dict)

if __name__ == '__main__':
    main()


