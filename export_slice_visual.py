import os
import numpy as np
import nibabel as nib
from glob import glob
import re
from tqdm import tqdm


class ImageAndMaskExporter:
    def __init__(self, base_dir, input_vis_dir, output_dir):
        """初始化图像和mask切片导出器"""
        self.base_dir = base_dir
        self.input_vis_dir = input_vis_dir
        self.output_dir = output_dir

        # 创建输出目录结构
        self.image_output_dir = os.path.join(output_dir, "images")
        self.mask_output_dir = os.path.join(output_dir, "masks")
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.mask_output_dir, exist_ok=True)

        # 模型列表
        self.model_names = [
            "YourModel", "nnUNet", "nnFormer",
            "MAML", "A2FSeg", "PA-Net",
            "mmFormer", "NestedFormer"
        ]

        # 模型特殊属性
        self.special_dir_models = ["mmFormer", "NestedFormer"]
        self.cross_fold_models = ["nnUNet", "nnFormer"]

        # mask语义标签（适配itksnap默认配色）
        self.label_mapping = {
            'background': 0,
            'incorrect': 1,  # 错误分割（假阳性）→ 红色
            'correct': 3,  # 正确分割 → 蓝色
            'false_negative': 2  # 假阴性（GT有但预测无）→ 绿色
        }

    def _get_case_dirs(self):
        """获取所有case文件夹"""
        case_pattern = os.path.join(self.input_vis_dir, "case_*")
        case_dirs = sorted(glob(case_pattern))
        return [d for d in case_dirs if os.path.isdir(d)]

    def _parse_case_info(self, case_dir):
        """解析病例信息（修复YNCH前缀处理）"""
        case_name = os.path.basename(case_dir)

        # 提取切片号
        slice_file = glob(os.path.join(case_dir, "*_slice_*.png"))[0]
        slice_z = int(re.search(r"slice_(\d+)\.png", os.path.basename(slice_file)).group(1))

        # 完整病例标识
        full_case_id = re.sub(r"case_\d+_", "", case_name)

        # 判断数据类型（内部/外部）
        data_type = "internal" if full_case_id.startswith("internal_") else "external"

        # 解析Task ID
        if data_type == "internal":
            # task_id = "Task131_DCET2Reg"
            task_id = "Task262_T2DCEReg"
        else:
            task_id = "Task263_externalValT2DCEReg" if "YNCH" in full_case_id else "Task264_CMUexternalValT2DCEReg"
            # task_id = "Task132_externalValDCET2Reg" if "YNCH" in full_case_id else "Task133_CMUexternalValDCET2Reg"

        # 修复：纯病例ID解析（保留YN2_/YN_前缀）
        if data_type == "internal":
            pure_case_id = re.sub(r"^internal_fold\d+_", "", full_case_id)
        else:
            if "YNCH" in full_case_id:
                pure_case_id = re.sub(r"^external_YNCH_", "", full_case_id)
            else:
                pure_case_id = re.sub(r"^external_\w+_", "", full_case_id)

        # 解析fold
        fold = int(re.search(r"fold(\d+)", full_case_id).group(1)) if data_type == "internal" else 0

        return {
            "full_case_id": full_case_id,
            "pure_case_id": pure_case_id,
            "slice_z": slice_z,
            "data_type": data_type,
            "task_id": task_id,
            "fold": fold
        }

    def _get_image_path(self, case_info):
        """获取原始图像路径（增强容错性）"""
        task_id = case_info["task_id"]
        pure_case_id = case_info["pure_case_id"]

        image_root = os.path.join(
            self.base_dir,
            f"MMFAFM_BCS/data/nnUNet_raw_data/{task_id}/images"
        )

        possible_patterns = [
            # f"{pure_case_id}.nii.gz",
            # f"{pure_case_id}_0000.nii.gz", # DCE
            f"{pure_case_id}_0001.nii.gz" # T2
        ]

        for pattern in possible_patterns:
            image_path = os.path.join(image_root, pattern)
            if os.path.exists(image_path):
                return image_path

        loose_matches = glob(os.path.join(image_root, f"{pure_case_id}*.nii.gz"))
        if loose_matches:
            return loose_matches[0]

        raise FileNotFoundError(
            f"未找到原始图像: {pure_case_id}（Task: {task_id}）\n"
            f"查找路径: {image_root}\n"
            f"尝试的模式: {possible_patterns}"
        )

    def _get_model_pred_path(self, model_name, case_info):
        """获取模型预测文件路径"""
        data_type = case_info["data_type"]
        pure_case_id = case_info["pure_case_id"]
        fold = case_info["fold"]

        model_configs = {
            "YourModel": {
                "internal": f"MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1",
                "external_ync": f"MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1",
                "external_cmu": f"MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"
            },
            "nnUNet": {
                "internal": f"nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1",
                "external_ync": f"nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1",
                "external_cmu": f"nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1"
            },
            "nnFormer": {
                "internal": f"nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1",
                "external_ync": f"nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1",
                "external_cmu": f"nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1"
            },
            "MAML": {
                "internal": f"MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1",
                "external_ync": f"MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1",
                "external_cmu": f"MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1"
            },
            "A2FSeg": {
                "internal": f"A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1",
                "external_ync": f"A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1",
                "external_cmu": f"A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"
            },
            "PA-Net": {
                "internal": f"PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1",
                "external_ync": f"PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1",
                "external_cmu": f"PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1"
            },
            "mmFormer": {
                "internal": f"mmFormer-main/pred_file/Task262_T2DCEReg",
                "external_ync": f"mmFormer-main/pred_file/Task263_externalValT2DCEReg",
                "external_cmu": f"mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg"
            },
            "NestedFormer": {
                "internal": f"NestedFormer-main/output_dir/Task262_T2DCEReg",
                "external_ync": f"NestedFormer-main/output_dir/Task263_externalValT2DCEReg",
                "external_cmu": f"NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg"
            }
        }

        if data_type == "internal":
            base_path = os.path.join(self.base_dir, model_configs[model_name]["internal"])
        else:
            source_key = "external_ync" if "YNCH" in case_info["full_case_id"] else "external_cmu"
            base_path = os.path.join(self.base_dir, model_configs[model_name][source_key])

        if model_name in self.special_dir_models:
            pred_path = os.path.join(base_path, f"fold_{fold}", f"{pure_case_id}.nii.gz")
        else:
            pred_path = os.path.join(base_path, f"fold_{fold}", "validation_raw", f"{pure_case_id}.nii.gz")

        if not os.path.exists(pred_path):
            pred_path = os.path.join(base_path, f"fold_{fold}", f"{pure_case_id}.nii.gz")

        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"模型 {model_name} 预测文件不存在: {pred_path}")

        return pred_path

    def _get_gt_path(self, case_info):
        """获取GT路径"""
        pure_case_id = case_info["pure_case_id"]
        fold = case_info["fold"]
        task_id = case_info["task_id"]

        gt_base = os.path.join(
            self.base_dir,
            f"MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/{task_id}/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"
        )
        gt_path = os.path.join(gt_base, f"fold_{fold}", "gt_niftis", f"{pure_case_id}.nii.gz")

        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_base, f"fold_{fold}", f"{pure_case_id}.nii.gz")
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"GT文件不存在: {gt_path}")

        return gt_path

    def _generate_overlay_mask(self, gt_slice, pred_slice):
        """生成叠加mask"""
        gt_binary = (gt_slice > 0.5).astype(bool)
        pred_binary = (pred_slice > 0.5).astype(bool)

        correct = np.logical_and(gt_binary, pred_binary)
        false_positive = np.logical_and(~gt_binary, pred_binary)
        false_negative = np.logical_and(gt_binary, ~pred_binary)

        mask = np.zeros_like(gt_slice, dtype=np.uint8)
        mask[false_positive] = self.label_mapping['incorrect']
        mask[correct] = self.label_mapping['correct']
        mask[false_negative] = self.label_mapping['false_negative']

        return mask

    def _export_image_slice(self, case_info, image_data, affine):
        """导出原始图像切片（X轴方向保留前2/3）"""
        full_case_id = case_info["full_case_id"]
        slice_z = case_info["slice_z"]

        # 提取切片并在X轴方向保留前2/3
        if len(image_data.shape) == 4:
            image_slice = image_data[:, :, slice_z, 0]  # 形状为 (Y, X)
        else:
            image_slice = image_data[:, :, slice_z]  # 形状为 (Y, X)

        # X轴方向保留前2/3
        x_length = image_slice.shape[1]
        x_cutoff = int(x_length * 2 / 3)  # 保留前2/3
        image_slice_cropped = image_slice[:, :x_cutoff]

        # 重塑为3D并调整 affine
        image_slice_3d = image_slice_cropped[:, :, np.newaxis]
        output_affine = affine.copy()
        output_affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * slice_z

        output_filename = f"{full_case_id}_slice_{slice_z}.nii.gz"
        output_path = os.path.join(self.image_output_dir, output_filename)
        nib.save(nib.Nifti1Image(image_slice_3d, output_affine), output_path)

        return output_path

    def _export_overlay_mask_slice(self, model_name, case_info, mask, affine, header):
        """导出叠加mask切片（X轴方向保留前2/3）"""
        full_case_id = case_info["full_case_id"]
        slice_z = case_info["slice_z"]

        # X轴方向保留前2/3
        x_length = mask.shape[1]
        x_cutoff = int(x_length * 2 / 3)  # 保留前2/3
        mask_cropped = mask[:, :x_cutoff]

        mask_3d = mask_cropped[:, :, np.newaxis]
        output_affine = affine.copy()
        output_affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * slice_z

        model_mask_dir = os.path.join(self.mask_output_dir, model_name)
        os.makedirs(model_mask_dir, exist_ok=True)

        output_filename = f"{full_case_id}_slice_{slice_z}_mask.nii.gz"
        output_path = os.path.join(model_mask_dir, output_filename)
        nib.save(nib.Nifti1Image(mask_3d, output_affine, header), output_path)

        return output_path

    def _export_gt_slice(self, case_info, gt_slice, affine, header):
        """导出GT切片（设置为绿色并在X轴方向保留前2/3）"""
        full_case_id = case_info["full_case_id"]
        slice_z = case_info["slice_z"]

        # 二值化并设置为绿色
        gt_binary = (gt_slice > 0.5).astype(np.uint8)
        gt_green = gt_binary * 2  # 绿色标签值为2

        # X轴方向保留前2/3
        x_length = gt_green.shape[1]
        x_cutoff = int(x_length * 2 / 3)  # 保留前2/3
        gt_cropped = gt_green[:, :x_cutoff]

        gt_3d = gt_cropped[:, :, np.newaxis]
        output_affine = affine.copy()
        output_affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * slice_z

        gt_dir = os.path.join(self.mask_output_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)

        output_filename = f"{full_case_id}_slice_{slice_z}_gt.nii.gz"
        output_path = os.path.join(gt_dir, output_filename)
        nib.save(nib.Nifti1Image(gt_3d, output_affine, header), output_path)

        return output_path

    def export_all(self):
        """导出所有病例的原始图像切片、叠加mask切片和GT切片"""
        for case_dir in tqdm(self._get_case_dirs(), desc="总进度"):
            try:
                case_info = self._parse_case_info(case_dir)
                full_case_id = case_info["full_case_id"]
                slice_z = case_info["slice_z"]
                print(f"\n处理病例: {full_case_id}, 切片: {slice_z}")

                # 导出原始图像
                image_path = self._get_image_path(case_info)
                image_img = nib.load(image_path)
                image_data = image_img.get_fdata()
                image_affine = image_img.affine
                image_path = self._export_image_slice(case_info, image_data, image_affine)
                print(f"已保存原始图像切片: {image_path}")

                # 加载GT
                gt_path = self._get_gt_path(case_info)
                gt_img = nib.load(gt_path)
                gt_data = gt_img.get_fdata()
                gt_slice = gt_data[:, :, slice_z]
                gt_affine = gt_img.affine
                gt_header = gt_img.header

                # 导出GT切片
                gt_slice_path = self._export_gt_slice(case_info, gt_slice, gt_affine, gt_header)
                print(f"已保存GT切片: {gt_slice_path}")

                # 导出各模型的mask
                for model_name in self.model_names:
                    try:
                        pred_path = self._get_model_pred_path(model_name, case_info)
                        pred_img = nib.load(pred_path)
                        pred_data = pred_img.get_fdata()
                        pred_slice = pred_data[:, :, slice_z]

                        overlay_mask = self._generate_overlay_mask(gt_slice, pred_slice)
                        mask_path = self._export_overlay_mask_slice(
                            model_name, case_info, overlay_mask, gt_affine, gt_header
                        )
                        print(f"已保存 {model_name} 叠加mask切片: {mask_path}")
                    except Exception as e:
                        print(f"处理模型 {model_name} 时出错: {str(e)}，跳过")
                        continue

            except Exception as e:
                print(f"处理病例 {case_dir} 时出错: {str(e)}，跳过")
                continue


if __name__ == "__main__":
    BASE_DIR = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2"
    INPUT_VIS_DIR = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/T2_segmentation_visualization_results"
    OUTPUT_DIR = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/T2_separate_image_mask_slices"

    exporter = ImageAndMaskExporter(BASE_DIR, INPUT_VIS_DIR, OUTPUT_DIR)
    exporter.export_all()
    print("\n所有原始图像切片、GT切片和叠加mask切片导出完成！")
