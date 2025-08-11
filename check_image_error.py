# import os
# import numpy as np
# from tqdm import tqdm
#
# # 设定包含 .npz 文件的文件夹路径
# npz_folder = "/home/lyq/Downloads"  # 请替换为实际的文件夹路径
#
# # 遍历文件夹及其子文件夹
# for root, dirs, files in os.walk(npz_folder):
#     for file in tqdm(files):
#         if file.endswith('.npz'):
#             npz_path = os.path.join(root, file)
#             try:
#                 # 加载 .npz 文件
#                 npz_data = np.load(npz_path)
#                 has_nan = False
#                 # 遍历 npz 文件中的每个数组
#                 for key in npz_data.files:
#                     array = npz_data[key]
#                     # 检查数组中是否存在 NaN 值
#                     if np.isnan(array).any():
#                         print(f"文件 {os.path.basename(npz_path)} 中的数组 {key} 存在 NaN 值")
#                         has_nan = True
#                 if not has_nan:
#                     print(f"文件 {os.path.basename(npz_path)} 不存在 NaN 值")
#
#             except Exception as e:
#                 print(f"读取文件 {os.path.basename(npz_path)} 时出现错误: {e}")

import numpy as np

# 加载 npz 文件
file_path = '/home/lyq/Downloads/ZJ_959.npz'
# file_path = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_preprocessed/Task262_T2DCEReg/nnUNetData_plans_v2.1_stage0/SY2_10403063.npz'
try:
    data = np.load(file_path)
    # 假设标签数组的键为 'data' 中的第三个通道，你可能需要根据实际情况修改
    label_array = data['data'][2]
    # 获取数组中不同的值及其出现的次数
    unique_values, counts = np.unique(label_array, return_counts=True)

    # 打印每个不同值及其出现的次数
    for value, count in zip(unique_values, counts):
        print(f"值 {value} 出现了 {count} 次。")

    # 打印值的类型数量
    print(f"值的类型数量为 {len(unique_values)}。")

except KeyError:
    print("在 npz 文件中未找到指定的数组，请检查数组键名。")
except Exception as e:
    print(f"读取文件时出现错误: {e}")
