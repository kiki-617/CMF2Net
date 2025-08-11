import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import matplotlib.pyplot as plt

from glob import glob
import SimpleITK as sitk
from tqdm import tqdm
import argparse

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def load_mask(mask_path):
    """加载3D掩码文件"""
    try:
        # 尝试使用SimpleITK加载
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        return mask_array
    except Exception as e:
        print(f"无法加载文件 {mask_path}: {e}")
        return None


def visualize_mask_slices(mask_array, output_dir, mask_name, figsize=(12, 8)):
    """可视化3D掩码的所有切片并保存"""
    os.makedirs(output_dir, exist_ok=True)
    num_slices = mask_array.shape[0]

    # 获取掩码中的唯一标签值
    unique_labels = np.unique(mask_array)
    unique_labels = unique_labels[unique_labels != 0]  # 排除背景

    # 为每个标签创建颜色映射
    colors = plt.cm.get_cmap('tab20', len(unique_labels) + 1)

    for slice_idx in tqdm(range(num_slices), desc=f"处理 {mask_name}"):
        slice_data = mask_array[slice_idx]

        # 如果当前切片没有非零标签，则跳过
        if np.sum(slice_data) == 0:
            continue

        plt.figure(figsize=figsize)

        # 绘制原始掩码
        plt.subplot(1, 1, 1)
        plt.title(f"{mask_name} - 切片 {slice_idx}")
        im = plt.imshow(slice_data, cmap='gray')

        # 添加颜色条显示标签值
        cbar = plt.colorbar(im)
        cbar.set_ticks(unique_labels)
        cbar.set_ticklabels([f'标签 {int(l)}' for l in unique_labels])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{mask_name}_slice_{slice_idx:04d}.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main(args):
    # 获取所有掩码文件
    mask_files = glob(os.path.join(args.input_dir, "*.nii.gz"))
    mask_files.extend(glob(os.path.join(args.input_dir, "*.nii")))
    mask_files.extend(glob(os.path.join(args.input_dir, "*.nrrd")))

    print(f"找到 {len(mask_files)} 个掩码文件")

    # 为每个掩码创建单独的输出目录
    for mask_path in mask_files:
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        if mask_name.endswith('.nii'):
            mask_name = mask_name[:-4]

        output_subdir = os.path.join(args.output_dir, mask_name)

        # 加载掩码
        mask_array = load_mask(mask_path)
        if mask_array is None:
            continue

        # 可视化并保存切片
        visualize_mask_slices(mask_array, output_subdir, mask_name)

    print(f"所有掩码切片已保存到 {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D医学图像掩码可视化工具')
    parser.add_argument('--input_dir', type=str, default='/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task133_CMUexternalValDCET2Reg/labels', help='输入掩码文件夹路径')
    parser.add_argument('--output_dir', type=str, default='/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task133_CMUexternalValDCET2Reg/label_slice', help='输出图像文件夹路径')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8], help='图像大小 (宽度, 高度)')

    args = parser.parse_args()
    main(args)