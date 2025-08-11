import os
import json
import random

# 训练数据文件夹路径
data_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task131_DCET2Reg/images'

# 获取所有训练数据文件
data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('_0000.nii.gz')]

# 打乱数据文件列表
random.shuffle(data_files)

# 计算每一折的数据数量
fold_size = len(data_files) // 5

# 初始化五折交叉验证的数据划分
folds = []
for i in range(5):
    start = i * fold_size
    end = start + fold_size if i < 4 else len(data_files)
    val_files = data_files[start:end]
    train_files = [f for f in data_files if f not in val_files]
    # 对 train 和 val 文件名列表进行排序
    train_files_sorted = sorted([train_file.split('/')[-1].replace('_0000.nii.gz', '') for train_file in train_files])
    val_files_sorted = sorted([val_file.split('/')[-1].replace('_0000.nii.gz', '') for val_file in val_files])
    folds.append({
        'train': train_files_sorted,
        'val': val_files_sorted
    })

# 保存为 JSON 文件
json_data = [folds[i] for i in range(5)]
with open('/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task131_DCET2Reg/splits_final.json',
          'w') as f:
    json.dump(json_data, f, indent=4)

print('五折交叉验证的 JSON 文件已生成：splits_final.json')
