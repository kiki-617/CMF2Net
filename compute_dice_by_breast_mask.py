import os
import re
import json
import nibabel as nib
import numpy as np


def calculate_dice(seg1, seg2):
    """计算两个分割掩码的Dice系数"""
    intersection = np.logical_and(seg1, seg2).sum()
    dice = (2. * intersection) / (seg1.sum() + seg2.sum())
    return dice


def extract_id(filename):
    """
    从文件名中提取核心标识符
    支持格式:
    - SY2_DCE_10941714 → SY2_10941714
    - SY2_10941714 → SY2_10941714
    - SY_10665957 → SY_10665957
    - ZJ_18 → ZJ_18
    - 10403063 → 10403063
    """
    # 尝试匹配 "字母+数字+任意字符+数字" 格式
    match = re.search(r'([A-Za-z]+[0-9]*)\D*(\d+)', filename)
    if match:
        prefix = match.group(1)  # 提取前缀，如SY2
        number = match.group(2)  # 提取数字部分
        return f"{prefix}_{number}"  # 组合成 SY2_10941714 格式

    # 回退到只匹配数字的情况
    match = re.search(r'\d+', filename)
    return match.group(0) if match else None


# 文件路径
breast_mask_folder = '/home/lyq/Desktop/省医云南三序列数据汇总/DCE_Breast_Mask'
tumor_label_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task132_externalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/gt_niftis'
predicted_tumor_mask_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task132_externalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw'
summary_json_path = '/data/RESULTS_FOLDER/nnUNet/1000/Task132_externalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json'

# 读取summary.json文件并提取所有测试文件路径
with open(summary_json_path, 'r') as f:
    summary = json.load(f)

# 提取所有测试文件的ID和对应的Dice值
summary_test_ids = {}
for result in summary['results']['all']:
    test_path = result.get('test', '')
    if test_path and '1' in result and 'Dice' in result['1']:
        filename = os.path.basename(test_path)
        file_id = extract_id(filename)
        if file_id:
            summary_test_ids[file_id] = {
                'dice': result['1']['Dice'],
                'filename': filename
            }

# 获取所有文件夹中的文件ID和对应的完整路径
breast_mask_files = {}
for f in os.listdir(breast_mask_folder):
    if f.endswith('.nii.gz'):
        file_id = extract_id(f)
        if file_id:
            breast_mask_files[file_id] = os.path.join(breast_mask_folder, f)

tumor_label_files = {}
for f in os.listdir(tumor_label_folder):
    if f.endswith('.nii.gz'):
        file_id = extract_id(f)
        if file_id:
            tumor_label_files[file_id] = os.path.join(tumor_label_folder, f)

predicted_tumor_files = {}
for f in os.listdir(predicted_tumor_mask_folder):
    if f.endswith('.nii.gz'):
        file_id = extract_id(f)
        if file_id:
            predicted_tumor_files[file_id] = os.path.join(predicted_tumor_mask_folder, f)

# 计算四个集合的交集（基于ID）
common_ids = set(breast_mask_files.keys()) & \
             set(tumor_label_files.keys()) & \
             set(predicted_tumor_files.keys()) & \
             set(summary_test_ids.keys())

print(f"找到 {len(common_ids)} 个共同文件（基于ID匹配）")

# 找出predicted_tumor_mask_folder中未被匹配的文件
unmatched_predicted_files = []
for file_id, file_path in predicted_tumor_files.items():
    if file_id not in common_ids:
        unmatched_predicted_files.append({
            'file_id': file_id,
            'filename': os.path.basename(file_path)
        })

# 记录形状不匹配的文件
shape_mismatch_files = []

# 存储原Dice指标和新计算的Dice指标
original_dice_scores = []
new_dice_scores = []
processed_files = []
processed_file_ids = []

for file_id in common_ids:
    # 获取各文件夹中的完整路径
    breast_mask_path = breast_mask_files[file_id]
    tumor_label_path = tumor_label_files[file_id]
    predicted_tumor_mask_path = predicted_tumor_files[file_id]
    original_filename = summary_test_ids[file_id]['filename']

    # 读取文件
    try:
        breast_mask = nib.load(breast_mask_path).get_fdata()
        tumor_label = nib.load(tumor_label_path).get_fdata()
        predicted_tumor_mask = nib.load(predicted_tumor_mask_path).get_fdata()

        # 检查形状是否匹配
        if breast_mask.shape != predicted_tumor_mask.shape:
            shape_mismatch_files.append({
                'filename': original_filename,
                'breast_mask_shape': breast_mask.shape,
                'predicted_tumor_mask_shape': predicted_tumor_mask.shape,
                'file_id': file_id
            })
            continue  # 跳过形状不匹配的文件，避免计算错误

        # 肿瘤mask和乳房mask做交集
        intersection_mask = np.logical_and(predicted_tumor_mask, breast_mask)

        # 计算新的Dice系数
        new_dice = calculate_dice(intersection_mask, tumor_label)
        new_dice_scores.append(new_dice)

        # 从summary.json中找出原Dice指标
        original_dice = summary_test_ids[file_id]['dice']
        original_dice_scores.append(original_dice)
        processed_files.append(original_filename)
        processed_file_ids.append(file_id)

    except Exception as e:
        print(f"无法读取文件 {file_id}: {e}")
        continue

# 打印所有形状不匹配的文件
print("\n形状不匹配的文件:")
if shape_mismatch_files:
    for file in shape_mismatch_files:
        print(f"- ID: {file['file_id']}, 文件名: {file['filename']}")
        print(f"  乳房掩码形状: {file['breast_mask_shape']}")
        print(f"  预测肿瘤掩码形状: {file['predicted_tumor_mask_shape']}")
else:
    print("没有发现形状不匹配的文件")

# 打印predicted_tumor_mask_folder中未被匹配的文件
print("\npredicted_tumor_mask_folder中未被匹配的文件:")
if unmatched_predicted_files:
    for file in unmatched_predicted_files:
        print(f"- ID: {file['file_id']}, 文件名: {file['filename']}")
else:
    print("predicted_tumor_mask_folder中所有文件都被匹配")

# 确保两个列表长度相同
assert len(original_dice_scores) == len(new_dice_scores), "原始Dice和新Dice数量不匹配"



# 计算总体统计信息
if original_dice_scores:
    avg_original = np.mean(original_dice_scores)
    avg_new = np.mean(new_dice_scores)
    avg_diff = avg_new - avg_original
    print("\n总体统计:")
    print(f"平均原始Dice: {avg_original:.4f}")
    print(f"平均新Dice: {avg_new:.4f}")
    print(f"平均差异: {avg_diff:.4f}")
