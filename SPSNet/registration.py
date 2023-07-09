from tqdm import tqdm
import ants
import logging
import os
import shutil

method = 'Rigid'

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
os.makedirs("logs", exist_ok=True)
handler = logging.FileHandler("logs/log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
 
logger.addHandler(handler)
logger.addHandler(console)

logger.info("start ANTs registration!")

n = 80

for index in tqdm(range(n)):
    logger.info(f"start{index}")
    fixed_image_path = os.path.join('CT_clean', f'{index}.nii')
    moving_image_path = os.path.join('ct_fake_right_direction', f'{index}.nii')
    to_apply_image_path = os.path.join('BFE-nii', f"{index}.nii")

    save_dir = os.path.join(f'{method}-results', str(index))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    fix_img = ants.image_read(fixed_image_path)
    move_img = ants.image_read(moving_image_path)
    to_apply_image = ants.image_read(to_apply_image_path)
    outs = ants.registration(fix_img, move_img, type_of_transform=method)

    reg_img = outs['warpedmovout']

    reg_img.set_direction(fix_img.direction)
    reg_img.set_origin(fix_img.origin)
    reg_img.set_spacing(fix_img.spacing)

    ants.image_write(reg_img, os.path.join(save_dir, 'fake_ct_generated_registration.nii.gz'))
    logger.info(f"finish 1 for {index}")
    move2fix_matrix = outs['fwdtransforms']
    appled_img = ants.apply_transforms(fix_img, to_apply_image, move2fix_matrix)
    ants.image_write(appled_img, os.path.join(save_dir, 'final_results_bfe_registration.nii.gz'))
    logger.info(f"finish 2 for {index}")
    # print(move2fix_matrix)
    # shutil.copy(move2fix_matrix, os.path.join(save_dir, 'fwdtransforms.nii.gz'))
    # ants.image_write(move2fix_matrix, os.path.join(save_dir, 'fwdtransforms.mat'))

    # fix2move_matrix = outs['invtransforms'][1]
    # print(fix2move_matrix)
    # shutil.copy(fix2move_matrix, os.path.join(save_dir, 'invtransforms.nii.gz'))
    # ants.image_write(fix2move_matrix, os.path.join(save_dir, 'invtransforms.mat'))
    logger.info(f"finish ANTs registration for index: {index}!")
    # break
logger.info(f"All tasks finished!")