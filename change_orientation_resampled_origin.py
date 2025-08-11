import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm


def resample_image_to_spacing(image, label, target_spacing=(1, 1, 1)):
    """
    将图像的 spacing 重采样到指定的新 spacing
    :param image: 输入的 SimpleITK 图像对象
    :param label: 输入的 SimpleITK 标签对象
    :param new_spacing: 新的 spacing，默认为 (1, 1, 1)
    :return: 重采样后的图像对象和标签对象
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [int(round(osz * osp / targ_sp)) for osz, osp, targ_sp in
                zip(original_size, original_spacing, target_spacing)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())

    # 设置输出图像的数据类型和原图像一致
    resample.SetOutputPixelType(image.GetPixelID())
    resample.SetInterpolator(sitk.sitkLinear)  # 对于图像，使用线性插值（可根据实际情况调整）
    resampled_image = resample.Execute(image)

    # 设置输出标签的数据类型和原标签一致
    resample.SetOutputPixelType(label.GetPixelID())
    # 对标签使用最近邻插值以保持标签值准确
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_label = resample.Execute(label)
    return resampled_image, resampled_label


def flip_image_based_on_direction(image):
    """
    根据图像的方向信息对图像数组进行翻转操作
    :param image: 输入的 SimpleITK 图像对象
    :return: 翻转后的图像对象
    """
    direction = image.GetDirection()
    array = sitk.GetArrayFromImage(image)

    if direction[0] < 0:
        array = np.flip(array, axis=2)
    if direction[4] < 0:
        array = np.flip(array, axis=1)
    if direction[8] < 0:
        array = np.flip(array, axis=0)

    # 将处理后的数组转换回 SimpleITK 图像对象
    new_image = sitk.GetImageFromArray(array)
    new_image.SetSpacing(image.GetSpacing())
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return new_image


def process_and_save_image_and_label(input_image_path, input_label_path, output_image_path, output_label_path):
    """
    处理图像和标签并保存
    :param input_image_path: 输入图像的文件路径
    :param input_label_path: 输入标签的文件路径
    :param output_image_path: 输出图像的文件路径
    :param output_label_path: 输出标签的文件路径
    """
    # 读取图像和标签
    image = sitk.ReadImage(input_image_path)
    label = sitk.ReadImage(input_label_path)

    # 重采样到 (1, 1, 1) 的 spacing
    resampled_image, resampled_label = resample_image_to_spacing(image, label)

    # 根据方向信息翻转图像和标签
    flipped_image = flip_image_based_on_direction(resampled_image)
    flipped_label = flip_image_based_on_direction(resampled_label)

    # 设置 origin 为 (0, 0, 0)
    flipped_image.SetOrigin((0, 0, 0))
    flipped_label.SetOrigin((0, 0, 0))

    # 保存处理后的图像和标签
    sitk.WriteImage(flipped_image, output_image_path)
    sitk.WriteImage(flipped_label, output_label_path)


def batch_process_images_and_labels(input_image_folder, input_label_folder, output_image_folder, output_label_folder):
    """
    批量处理指定文件夹中的图像和标签
    :param input_image_folder: 输入图像所在的文件夹路径
    :param input_label_folder: 输入标签所在的文件夹路径
    :param output_image_folder: 输出图像保存的文件夹路径
    :param output_label_folder: 输出标签保存的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # 遍历输入图像文件夹中的所有文件
    for filename in tqdm(os.listdir(input_image_folder)):
        if filename.endswith('.nii'):
            input_image_path = os.path.join(input_image_folder, filename)

            label_filename = filename.replace('.nii','_mask.nii')
            input_label_path = os.path.join(input_label_folder, label_filename)

            output_image_path = os.path.join(output_image_folder, filename+'.gz')
            output_label_path = os.path.join(output_label_folder, label_filename+'.gz')

            # 检查输出文件是否已存在
            if os.path.exists(output_image_path) and os.path.exists(output_label_path):
                # print(f"Output files {output_image_path} and {output_label_path} already exist. Skipping.")
                continue

            if os.path.exists(input_label_path):
                try:
                    process_and_save_image_and_label(input_image_path, input_label_path, output_image_path,
                                                     output_label_path)
                    # print(f"Successfully processed {input_image_path} and {input_label_path}, saved to {output_image_path} and {output_label_path}")
                except Exception as e:
                    print(f"Error processing {input_image_path} and {input_label_path}: {e}")
            else:
                print(f"Label file {input_label_path} not found. Skipping.")


if __name__ == "__main__":
    input_image_folder = '/home/lyq/Desktop/中国医科大基线2+3/T2/image'
    input_label_folder = '/home/lyq/Desktop/中国医科大基线2+3/T2/tumor_mask'
    output_image_folder = '/home/lyq/Desktop/中国医科大基线2+3/T2/image_res'
    output_label_folder = '/home/lyq/Desktop/中国医科大基线2+3/T2/tumor_mask_res'
    batch_process_images_and_labels(input_image_folder, input_label_folder, output_image_folder, output_label_folder)
