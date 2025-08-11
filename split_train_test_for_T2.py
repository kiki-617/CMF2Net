import os
import shutil
import random
from tqdm import tqdm


def get_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nii.gz')]


def split_files(files, ratio=1):
    random.shuffle(files)
    train_size = int(len(files) * ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    return train_files, test_files


def get_file_number(file_name):
    # 提取文件名中的编号部分
    if '_DCE_' in file_name:
        parts = file_name.split('_DCE_')
    elif '_T2_' in file_name:
        parts = file_name.split('_T2_')
    else:
        return None
    return parts[1].split('.')[0]


def process_and_save_files(dce_files, t2_files, label_folders, train_image_dir, train_label_dir, test_image_dir,
                           test_label_dir):
    # 对 DCE 文件进行排序
    dce_files.sort()

    all_label_files = []
    for label_folder in label_folders:
        all_label_files.extend(get_files(label_folder))

    # 按编号配对 DCE 和 T2 文件
    paired_files = []
    for dce_file in dce_files:
        dce_number = get_file_number(os.path.basename(dce_file))
        for t2_file in t2_files:
            t2_number = get_file_number(os.path.basename(t2_file))
            if dce_number == t2_number:
                paired_files.append((dce_file, t2_file))
                break

    train_pairs, test_pairs = split_files(paired_files)
    train_files = [file for pair in train_pairs for file in pair]
    test_files = [file for pair in test_pairs for file in pair]

    def process_subset(files, image_dir, label_dir):
        for file in tqdm(files):
            file_name = os.path.basename(file)

            # 去除文件名中的 _DCE_ 或 _T2_
            if 'DCE' in file_name:
                base_part = file_name.replace('_DCE_', '_')  # 将 SY2_DCE_123 转为 SY2_123
                new_file_name = base_part.replace('.nii.gz', '_0001.nii.gz')
            elif 'T2' in file_name:
                base_part = file_name.replace('_T2_', '_')  # 将 SY2_T2_123 转为 SY2_123
                new_file_name = base_part.replace('.nii.gz', '_0000.nii.gz')

            target_image_path = os.path.join(image_dir, new_file_name)
            if not os.path.exists(target_image_path):
                shutil.copy(file, target_image_path)

            # 查找对应的标签文件（基于处理后的文件名基础部分和编号）
            file_number = get_file_number(file_name)
            for label_file in all_label_files:
                label_file_number = get_file_number(os.path.basename(label_file))
                if label_file_number == file_number:
                    label_base = os.path.basename(label_file)
                    if 'DCE' in label_base:
                        label_base = label_base.replace('_DCE_', '_')
                    if 'T2' in label_base:
                        label_base = label_base.replace('_T2_', '_')
                    target_label_path = os.path.join(label_dir, label_base)
                    if not os.path.exists(target_label_path):
                        shutil.copy(label_file, target_label_path)

    # 处理训练集
    process_subset(train_files, train_image_dir, train_label_dir)

    # 处理测试集
    process_subset(test_files, test_image_dir, test_label_dir)


# 源文件夹路径
dce_folder_1 = '/home/lyq/Desktop/YN外部验证/reference_T2/T2_res'
t2_folder_1 = '/home/lyq/Desktop/YN外部验证/reference_T2/DCE_reg'
label_folder_1 = '/home/lyq/Desktop/YN外部验证/reference_T2/T2_tumor_label_res'

# dce_folder_2 = '/home/lyq/Desktop/SY+ZJ（内部）/ZJ/reference_T2/T2_res'
# t2_folder_2 = '/home/lyq/Desktop/SY+ZJ（内部）/ZJ/reference_T2/DCE_reg'
# label_folder_2 = '/home/lyq/Desktop/SY+ZJ（内部）/ZJ/reference_T2/T2_tumor_label_res'

# 目标文件夹路径（注意：nnUNet要求训练图像文件夹为imagesTr，之前代码中路径有误，已修正）
train_image_dir = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task263_externalValT2DCEReg/images'
train_label_dir = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task263_externalValT2DCEReg/labels'
test_image_dir = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task263_externalValT2DCEReg/imagesTs'
test_label_dir = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task263_externalValT2DCEReg/labelsTs'

# 创建目标文件夹（修正nnUNet标准路径：训练图像应为imagesTr）
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# 获取所有文件
# dce_files_all = get_files(dce_folder_1) + get_files(dce_folder_2)
# t2_files_all = get_files(t2_folder_1) + get_files(t2_folder_2)
dce_files_all = get_files(dce_folder_1)
t2_files_all = get_files(t2_folder_1)

# 设置随机数种子，保证划分的唯一性
random.seed(42)

# 处理和保存文件
process_and_save_files(dce_files_all, t2_files_all, [label_folder_1], train_image_dir, train_label_dir,
# process_and_save_files(dce_files_all, t2_files_all, [label_folder_1, label_folder_2], train_image_dir, train_label_dir,
                       test_image_dir, test_label_dir)
