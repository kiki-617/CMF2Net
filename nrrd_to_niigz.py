import os
import SimpleITK as sitk

# 定义输入文件夹路径
from tqdm import tqdm

input_folder = "/home/lyq/Desktop/浙江肿瘤乳腺-张臻/breast cancer/train pre dce/train pre dce"

# 定义输出文件夹路径
image_output_folder = "/home/lyq/Desktop/浙肿数据/DCE"
tumor_output_folder = "/home/lyq/Desktop/浙肿数据/DCE_tumor_mask"
lymph_node_output_folder = "/home/lyq/Desktop/浙肿数据/DCE_lymph_mask"

# 创建输出文件夹（如果不存在）
os.makedirs(tumor_output_folder, exist_ok=True)
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(lymph_node_output_folder, exist_ok=True)

# 遍历输入文件夹中的文件
for filename in tqdm(os.listdir(input_folder)):
    file_path = os.path.join(input_folder, filename)

    if os.path.isfile(file_path) and filename.endswith('.nrrd'):
        # 读取 nrrd 文件
        image = sitk.ReadImage(file_path)

        # 提取原文件编号
        file_number = filename.split("_")[0]

        # 根据文件名判断文件类型并保存到相应的输出文件夹
        if "_seg1.seg.nrrd" in filename:
            # 肿瘤标签
            output_filename = f"ZJ_DCE_{file_number}.nii.gz"
            output_path = os.path.join(tumor_output_folder, output_filename)
            sitk.WriteImage(image, output_path)
        elif "_dce.nrrd" in filename:
            # 图像
            output_filename = f"ZJ_DCE_{file_number}.nii.gz"
            output_path = os.path.join(image_output_folder, output_filename)
            sitk.WriteImage(image, output_path)
        elif "_seg2.seg.nrrd" in filename:
            # 淋巴结标签
            output_filename = f"ZJ_DCE_{file_number}.nii.gz"
            output_path = os.path.join(lymph_node_output_folder, output_filename)
            sitk.WriteImage(image, output_path)

print("转换完成！")