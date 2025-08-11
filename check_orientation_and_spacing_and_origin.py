import os
import SimpleITK as sitk
from tqdm import tqdm

# 定义目标 Spacing 和 Orientation
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # RAI 方向
TARGET_ORIGIN = (0.0, 0.0, 0.0)

# 定义需要检查的文件夹路径
folders_to_check = [
   '/home/lyq/Desktop/SY+ZJ（内部）/ZJ/DCE_res',
    '/home/lyq/Desktop/SY+ZJ（内部）/ZJ/DCE_tumor_label_res',

]

def check_spacing_and_orientation(image_path):
    """
    检查图像的 Spacing、Orientation 和 Origin 是否符合要求
    :param image_path: 图像文件路径
    :return: (是否 Spacing 正确, 是否 Orientation 正确, 是否 Origin 正确)
    """
    # 读取图像
    image = sitk.ReadImage(image_path)

    # 检查 Spacing
    spacing = image.GetSpacing()
    is_spacing_correct = spacing == TARGET_SPACING

    # 检查 Orientation
    direction = image.GetDirection()
    is_orientation_correct = direction == TARGET_ORIENTATION

    # 检查 Origin
    origin = image.GetOrigin()
    is_origin_correct = origin == TARGET_ORIGIN

    return is_spacing_correct, is_orientation_correct, is_origin_correct

def check_folders(folders):
    """
    检查文件夹中的所有图像文件
    :param folders: 需要检查的文件夹列表
    """
    for folder in folders:
        print(f"Checking folder: {folder}")
        for root, _, files in os.walk(folder):
            for file in tqdm(files):
                if file.endswith('.nii.gz') or file.endswith('.nii'):  # 支持 .nii.gz 和 .nii 文件
                    image_path = os.path.join(root, file)
                    is_spacing_correct, is_orientation_correct, is_origin_correct = check_spacing_and_orientation(image_path)

                    # 输出检查结果
                    if not is_spacing_correct or not is_orientation_correct or not is_origin_correct:
                        print(f"File: {image_path}")
                        if not is_spacing_correct:
                            print(f"  - Spacing is incorrect: {sitk.ReadImage(image_path).GetSpacing()}")
                        if not is_orientation_correct:
                            print(f"  - Orientation is incorrect: {sitk.ReadImage(image_path).GetDirection()}")
                        if not is_origin_correct:
                            print(f"  - Origin is incorrect: {sitk.ReadImage(image_path).GetOrigin()}")
                    else:
                        print(f"File: {image_path} - Spacing, Orientation and Origin are correct.")

# 检查所有文件夹
check_folders(folders_to_check)
