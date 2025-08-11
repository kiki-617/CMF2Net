import os

# 定义图像文件夹和标签文件夹的路径
image_folder = "/home/lyq/Desktop/SY+ZJ（内部）/SY/reference_T2/DCE_reg"
# label_folder = "/home/lyq/Desktop/YN外部验证/DCE_tumor_label_res"

# 遍历图像文件夹中的文件
for image_name in os.listdir(image_folder):
    if image_name.endswith("_0000.nii.gz"):
        image_file = os.path.join(image_folder, image_name)
        new_image_name = image_name.replace("_0000.nii.gz", ".nii.gz")
        new_image_file = os.path.join(image_folder, new_image_name)

        try:
            os.rename(image_file, new_image_file)
            print(f"图像文件 {image_file} 已重命名为 {new_image_file}")
        except FileNotFoundError:
            print(f"未找到图像文件: {image_file}")
        except Exception as e:
            print(f"重命名图像文件 {image_file} 时出现错误: {e}")

# # 遍历标签文件夹中的文件
# for label_name in os.listdir(label_folder):
#     if label_name.endswith("_gt_resampled.nii.gz"):
#         label_file = os.path.join(label_folder, label_name)
#         new_label_name = label_name.replace("_gt_resampled.nii.gz", ".nii.gz")
#         new_label_file = os.path.join(label_folder, new_label_name)
#
#         try:
#             os.rename(label_file, new_label_file)
#             print(f"标签文件 {label_file} 已重命名为 {new_label_file}")
#         except FileNotFoundError:
#             print(f"未找到标签文件: {label_file}")
#         except Exception as e:
#             print(f"重命名标签文件 {label_file} 时出现错误: {e}")
