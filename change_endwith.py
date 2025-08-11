import os
import re
from pathlib import Path

from tqdm import tqdm


def batch_swap_file_extensions(directory, pattern1, pattern2):
    """
    批量交换符合特定模式的文件扩展名

    参数:
    directory (str): 文件所在目录路径
    pattern1 (str): 第一个模式，例如'*_0000.nii.gz'
    pattern2 (str): 第二个模式，例如'*_0001.nii.gz'
    """
    # 获取匹配两种模式的文件列表
    path = Path(directory)
    files_pattern1 = list(path.glob(pattern1))
    files_pattern2 = list(path.glob(pattern2))

    # 创建映射字典，用于匹配对应的文件对
    file_mapping = {}

    # 提取基础文件名作为键，用于匹配
    for file1 in files_pattern1:
        base_name = file1.name.rsplit('_', 1)[0]
        file_mapping[base_name] = (file1, None)

    # 查找匹配的文件对
    for file2 in files_pattern2:
        base_name = file2.name.rsplit('_', 1)[0]
        if base_name in file_mapping:
            file_mapping[base_name] = (file_mapping[base_name][0], file2)

    # 执行重命名操作
    for base_name, (file1, file2) in tqdm(file_mapping.items()):
        if file1 and file2:
            try:
                # 创建临时文件名
                temp1 = file1.with_suffix('.temp')
                temp2 = file2.with_suffix('.temp')

                # 执行重命名操作（使用临时文件避免冲突）
                file1.rename(temp1)
                file2.rename(file1)
                temp1.rename(file2)

                print(f"已交换: {file1.name} <-> {file2.name}")
            except Exception as e:
                print(f"处理文件对 {file1.name} 和 {file2.name} 时出错: {str(e)}")


if __name__ == "__main__":
    # 配置参数
    directory = "/nas/liuyaoqi/projects/MMFAFM_BCS/data/nnUNet_raw_data/Task262_T2DCEReg/images"
    directory_ts = "/nas/liuyaoqi/projects/MMFAFM_BCS/data/nnUNet_raw_data/Task262_T2DCEReg/imagesTs"
    pattern1 = "*_0000.nii.gz"
    pattern2 = "*_0001.nii.gz"

    # 执行批量重命名
    batch_swap_file_extensions(directory, pattern1, pattern2)
    batch_swap_file_extensions(directory_ts, pattern1, pattern2)