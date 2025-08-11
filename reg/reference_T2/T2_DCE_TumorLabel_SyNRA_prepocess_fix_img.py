import shutil
import numpy as np
import os.path
import ants
from glob import glob
from tqdm import tqdm
from os.path import join, basename
from monai.metrics import DiceMetric
import torch

# DCE序列的图像和标签文件夹路径
DCE_imageTs_path = "/home/lyq/Desktop/中国医科大基线2+3/T2/image_res"
DCE_labelTs_path = "/home/lyq/Desktop/中国医科大基线2+3/T2/tumor_mask_res"
# T2序列的图像和标签文件夹路径
T2_imageTs_path = "/home/lyq/Desktop/中国医科大基线2+3/DCE/image_res"
T2_labelTs_path = "/home/lyq/Desktop/中国医科大基线2+3/DCE/tumor_mask_res"

T2_warped_imagesTs_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/CMU/SyNRA/DCE/warped_images"
T2_warped_labelsTs_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/CMU/SyNRA/DCE/warped_labels"
T2_inversed_transTs_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/CMU/SyNRA/DCE/inversed_trans"
DCE_inversed_labelsTs_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/CMU/SyNRA/DCE/inversed_labels"
os.makedirs(T2_warped_imagesTs_path, exist_ok=True)
os.makedirs(T2_warped_labelsTs_path, exist_ok=True)
os.makedirs(T2_inversed_transTs_path, exist_ok=True)
os.makedirs(DCE_inversed_labelsTs_path, exist_ok=True)


def get_number_from_filename(filename):
    """从图像文件名中提取序号（假设序号在倒数第二个位置，以'_'分割）"""
    parts = filename.split('_')
    return parts[2].replace('.nii.gz', '')

def get_number_from_label_filename(filename):
    """从图像文件名中提取序号（假设序号在倒数第二个位置，以'_'分割）"""
    parts = filename.split('_')
    return parts[3].replace('.nii.gz', '')


# 获取DCE序列图像和标签文件的序号集合
DCE_imageTs_numbers = {get_number_from_filename(x) for x in glob(join(DCE_imageTs_path, "*.nii.gz"))}
DCE_labelTs_numbers = {get_number_from_label_filename(x) for x in glob(join(DCE_labelTs_path, "*.nii.gz"))}
# 获取T2序列图像和标签文件的序号集合
T2_imageTs_numbers = {get_number_from_filename(x) for x in glob(join(T2_imageTs_path, "*.nii.gz"))}
T2_labelTs_numbers = {get_number_from_label_filename(x) for x in glob(join(T2_labelTs_path, "*.nii.gz"))}

# 找到所有文件序号的交集
common_numbers = DCE_imageTs_numbers.intersection(DCE_labelTs_numbers).intersection(T2_imageTs_numbers).intersection(T2_labelTs_numbers)

# 根据交集序号获取对应的文件列表
DCE_imageTs_files = [x for x in glob(join(DCE_imageTs_path, "*.nii.gz")) if get_number_from_filename(x) in common_numbers]
T2_imageTs_files = [x for x in glob(join(T2_imageTs_path, "*.nii.gz")) if get_number_from_filename(x) in common_numbers]
DCE_labelTs_files = [x for x in glob(join(DCE_labelTs_path, "*.nii.gz")) if get_number_from_label_filename(x) in common_numbers]
T2_labelTs_files = [x for x in glob(join(T2_labelTs_path, "*.nii.gz")) if get_number_from_label_filename(x) in common_numbers]

# 对文件列表按照序号进行排序
DCE_imageTs_files = sorted(DCE_imageTs_files, key=lambda x: get_number_from_filename(x))
T2_imageTs_files = sorted(T2_imageTs_files, key=lambda x: get_number_from_filename(x))
DCE_labelTs_files = sorted(DCE_labelTs_files, key=lambda x: get_number_from_label_filename(x))
T2_labelTs_files = sorted(T2_labelTs_files, key=lambda x: get_number_from_label_filename(x))

# 临时文件路径
temp_result_file_path = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/ZJ/SyNRA/DCE/dice_results_temp.txt'
final_result_file_path = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_T2/ZJ/SyNRA/DCE/dice_results_train.txt'

# 打开临时文件用于写入结果
temp_result_file = open(temp_result_file_path, 'w')
temp_result_file.write("File Name\tDice Coefficient\n")

dice_scores = []
dice_metric = DiceMetric(include_background=True, reduction="mean")

# 训练+验证集（按照文件名中的序号进行排序）文件进行处理
for DCE_imageTs_file, T2_imageTs_file, DCE_labelTs_file, T2_labelTs_file in \
        tqdm(zip(DCE_imageTs_files, T2_imageTs_files, DCE_labelTs_files, T2_labelTs_files)):
    # 获取图像文件名中的序号（使用之前定义的提取序号的函数，这里假设函数已正确定义）
    DCE_image_number = get_number_from_filename(DCE_imageTs_file)
    T2_image_number = get_number_from_filename(T2_imageTs_file)

    DCE_label_number = get_number_from_label_filename(DCE_labelTs_file)
    T2_label_number = get_number_from_label_filename(T2_labelTs_file)

    # 使用assert语句确保序号一致
    assert DCE_image_number == T2_image_number == DCE_label_number, f"{basename(DCE_imageTs_file)}图像文件和{basename(T2_imageTs_file)}序号对不上"

    # 检查输出文件是否已经存在
    warped_image_T2_path = join(T2_warped_imagesTs_path,
                                basename(T2_imageTs_file).replace('.nii.gz', '.nii.gz'))
    warped_label_T2_path = join(T2_warped_labelsTs_path,
                                basename(T2_labelTs_file).replace('.nii.gz', '.nii.gz'))
    inversed_trans_path = join(T2_inversed_transTs_path,
                               basename(T2_imageTs_file).replace('.nii.gz', '_inversed.mat'))
    inversed_DCE_label_path = join(DCE_inversed_labelsTs_path,
                                   basename(T2_labelTs_file).replace('.nii.gz', '_gt_inversed.nii.gz'))

    if os.path.exists(warped_image_T2_path) and os.path.exists(warped_label_T2_path) and \
            os.path.exists(inversed_trans_path) and os.path.exists(inversed_DCE_label_path):
        print(f"跳过 {basename(T2_imageTs_file)}，输出文件已存在。")
        continue

    # 读取DCE序列的图像和标签
    img_DCE = ants.image_read(DCE_imageTs_file)
    label_DCE = ants.image_read(DCE_labelTs_file)

    # 读取T2序列的图像和标签
    img_T2 = ants.image_read(T2_imageTs_file)
    label_T2 = ants.image_read(T2_labelTs_file)

    ants_T2 = ants.registration(fixed=label_DCE, moving=label_T2, type_of_transform='SyNRA')
    warped_image_T2 = ants.apply_transforms(fixed=img_DCE, moving=img_T2, transformlist=ants_T2['fwdtransforms'],
                                            interpolator="linear")
    warped_label_T2 = ants.apply_transforms(fixed=label_DCE, moving=label_T2, transformlist=ants_T2['fwdtransforms'],
                                            interpolator="nearestNeighbor")

    inversed_DCE_label = ants.apply_transforms(fixed=label_T2, moving=label_DCE, transformlist=ants_T2['invtransforms'],
                                               interpolator="nearestNeighbor")

    # 保存配准后的T2配准标签
    ants.image_write(warped_image_T2, warped_image_T2_path)
    ants.image_write(warped_label_T2, warped_label_T2_path)
    shutil.copy(ants_T2['invtransforms'][0], inversed_trans_path)
    ants.image_write(inversed_DCE_label, inversed_DCE_label_path)

    # 计算Dice系数
    label_T2_np = label_T2.numpy()[np.newaxis, np.newaxis, ...]
    inversed_DCE_label_np = inversed_DCE_label.numpy()[np.newaxis, np.newaxis, ...]
    label_T2_tensor = torch.from_numpy(label_T2_np).float()
    inversed_label_DCE_tensor = torch.from_numpy(inversed_DCE_label_np).float()
    dice = dice_metric(y_pred=inversed_label_DCE_tensor, y=label_T2_tensor).item()
    dice_scores.append(dice)

    # 写入临时文件
    temp_result_file.write(f"{basename(T2_imageTs_file)}\t{dice}\n")

# 关闭临时文件
temp_result_file.close()

# 计算平均Dice系数
average_dice = np.mean(dice_scores)
print(f"Average Dice score: {average_dice}")

# 将平均Dice值和临时文件内容写入最终结果文件
with open(final_result_file_path, 'w') as final_result_file:
    final_result_file.write(f"Average Dice score: {average_dice}\n\n")
    with open(temp_result_file_path, 'r') as temp_file:
        final_result_file.write(temp_file.read())

# 删除临时文件
os.remove(temp_result_file_path)
