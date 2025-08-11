import pickle
import numpy as np

import torch
checkpoint = torch.load('/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/'
                        'nnUNet/3d_fullres/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/model_best.model', map_location='cpu')  # 加载模型参数

print(checkpoint['epoch'])  # 打印参数结构
# 1494
# 1473
# 1407
# 1480
# 1496


# # 假设pkl文件路径
# pkl_file_path = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF/RESULTS_FOLDER/nnUNet/3d_fullres/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/model_best.model.pkl'
#
# try:
#     with open(pkl_file_path, 'rb') as f:
#         data = pickle.load(f)
#         print(data)
# except FileNotFoundError:
#     print(f"文件 {pkl_file_path} 不存在。")
# except:
#     print(f"读取 {pkl_file_path} 时出现错误。")
#
# print("========================================================================================================================================================================================================================================")
# print("========================================================================================================================================================================================================================================")
# # 假设npz文件路径
# npz_file_path = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_preprocessed/Task083_debugBraTS2020/nnUNetData_plans_v2.1_stage0/BraTS20_Training_001.npz'
#
# try:
#     with np.load(npz_file_path) as data:
#         for key in data.keys():
#             print(f"键: {key}, 值: {data[key].shape}")
# except FileNotFoundError:
#     print(f"文件 {npz_file_path} 不存在。")
# except:
#     print(f"读取 {npz_file_path} 时出现错误。")
