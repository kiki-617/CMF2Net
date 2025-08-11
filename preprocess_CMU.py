import os
import re
import shutil
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_rename.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description='批量重命名和移动医学图像文件')
    parser.add_argument('--data_dir', default='/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/nnUNet_raw_data/Task264_CMUexternalValT2DCEReg',
                        help='数据根目录，例如/media/lyq/.../Task133_CMUexternalValDCET2Reg')
    parser.add_argument('--dce_dir', default='warped_images',
                        help='DCE图像所在目录，相对于data_dir')
    parser.add_argument('--t2_dir', default='image_res',
                        help='T2图像所在目录，相对于data_dir')
    parser.add_argument('--labels_dir', default='tumor_mask_res',
                        help='标签所在目录，相对于data_dir')
    parser.add_argument('--output_images_dir', default='images',
                        help='输出图像目录，相对于data_dir')
    parser.add_argument('--output_labels_dir', default='labels',
                        help='输出标签目录，相对于data_dir')
    parser.add_argument('--overwrite', action='store_true',
                        help='是否覆盖已存在的文件')
    return parser.parse_args()


def extract_patient_id(filename):
    """从文件名中提取患者ID

    例如:
    DCE_ANLI20220429.nii.gz -> ANLI20220429
    T2_ANLI20220429.nii.gz -> ANLI20220429
    DCE_ANLI20220429_mask.nii.gz -> ANLI20220429
    """
    # 使用正则表达式匹配模式
    patterns = [
        r'^(?:DCE|T2)_(.+?)(?:_mask)?\.nii\.gz$',  # 匹配标准模式
        r'^(.+?)_(?:DCE|T2)(?:_mask)?\.nii\.gz$',  # 匹配另一种可能的模式
    ]

    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1)

    # 如果没有匹配到，尝试其他启发式方法
    parts = filename.split('_')
    if len(parts) >= 2:
        if parts[0] in ['DCE', 'T2']:
            # 处理 DCE_ANLI20220429_mask.nii.gz 这种情况
            potential_id = parts[1]
            if potential_id.endswith('_mask'):
                potential_id = potential_id[:-5]
            return potential_id
        else:
            # 尝试其他部分
            for part in parts:
                if re.search(r'\d{8}', part):  # 查找包含8位数字的部分
                    return part

    logger.warning(f"无法从文件名 {filename} 中提取患者ID")
    return None


def process_files(args):
    """处理所有文件"""
    # 创建输出目录
    output_images_path = Path(args.data_dir) / args.output_images_dir
    output_labels_path = Path(args.data_dir) / args.output_labels_dir

    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)

    # 获取DCE和T2图像文件列表
    dce_dir_path = Path(args.data_dir) / args.dce_dir
    t2_dir_path = Path(args.data_dir) / args.t2_dir
    labels_dir_path = Path(args.data_dir) / args.labels_dir

    # 确保目录存在
    if not dce_dir_path.exists():
        logger.error(f"DCE目录不存在: {dce_dir_path}")
        return

    if not t2_dir_path.exists():
        logger.error(f"T2目录不存在: {t2_dir_path}")
        return

    if not labels_dir_path.exists():
        logger.error(f"标签目录不存在: {labels_dir_path}")
        return

    # 获取所有文件
    dce_files = [f for f in dce_dir_path.glob('*.nii.gz') if f.is_file()]
    t2_files = [f for f in t2_dir_path.glob('*.nii.gz') if f.is_file()]
    label_files = [f for f in labels_dir_path.glob('*.nii.gz') if f.is_file()]

    logger.info(f"找到 {len(dce_files)} 个DCE图像文件")
    logger.info(f"找到 {len(t2_files)} 个T2图像文件")
    logger.info(f"找到 {len(label_files)} 个标签文件")

    # 创建患者ID到文件的映射
    dce_map = {extract_patient_id(f.name): f for f in dce_files}
    t2_map = {extract_patient_id(f.name): f for f in t2_files}
    label_map = {extract_patient_id(f.name): f for f in label_files}

    # 获取所有患者ID
    all_patient_ids = set(dce_map.keys()).union(t2_map.keys()).union(label_map.keys())
    logger.info(f"总共发现 {len(all_patient_ids)} 个患者ID")

    # 处理每个患者
    success_count = 0
    failed_count = 0

    for patient_id in tqdm(all_patient_ids, desc="处理患者"):
        if not patient_id:
            failed_count += 1
            continue

        try:
            # 处理DCE图像
            if patient_id in dce_map:
                dce_file = dce_map[patient_id]
                new_dce_name = f"{patient_id}_0000.nii.gz"
                new_dce_path = output_images_path / new_dce_name

                if new_dce_path.exists() and not args.overwrite:
                    logger.warning(f"DCE文件已存在，跳过: {new_dce_path}")
                else:
                    shutil.move(dce_file, new_dce_path)
                    logger.info(f"移动 {dce_file} 到 {new_dce_path}")

            # 处理T2图像
            if patient_id in t2_map:
                t2_file = t2_map[patient_id]
                new_t2_name = f"{patient_id}_0001.nii.gz"
                new_t2_path = output_images_path / new_t2_name

                if new_t2_path.exists() and not args.overwrite:
                    logger.warning(f"T2文件已存在，跳过: {new_t2_path}")
                else:
                    shutil.move(t2_file, new_t2_path)
                    logger.info(f"移动 {t2_file} 到 {new_t2_path}")

            # 处理标签
            if patient_id in label_map:
                label_file = label_map[patient_id]
                new_label_name = f"{patient_id}.nii.gz"
                new_label_path = output_labels_path / new_label_name

                if new_label_path.exists() and not args.overwrite:
                    logger.warning(f"标签文件已存在，跳过: {new_label_path}")
                else:
                    shutil.move(label_file, new_label_path)
                    logger.info(f"移动 {label_file} 到 {new_label_path}")

            success_count += 1
        except Exception as e:
            failed_count += 1
            logger.error(f"处理患者 {patient_id} 时出错: {str(e)}")

    logger.info(f"处理完成: 成功={success_count}, 失败={failed_count}")


if __name__ == "__main__":
    args = setup_argparse()
    process_files(args)