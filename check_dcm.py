import pydicom
import os


def print_dicom_info(dicom_path):
    """
    读取DICOM文件并打印其元数据信息

    参数:
        dicom_path: DICOM文件的路径
    """
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(dicom_path)
        print(ds)
        print(f"成功读取DICOM文件: {dicom_path}\n")
        print(f"患者信息:")
        print(f"  患者ID: {ds.PatientID if 'PatientID' in ds else 'N/A'}")
        print(f"  患者姓名: {ds.PatientName if 'PatientName' in ds else 'N/A'}")
        print(f"  患者性别: {ds.PatientSex if 'PatientSex' in ds else 'N/A'}")
        print(f"  患者年龄: {ds.PatientAge if 'PatientAge' in ds else 'N/A'}")
        print(f"  患者出生日期: {ds.PatientBirthDate if 'PatientBirthDate' in ds else 'N/A'}\n")

        print(f"检查信息:")
        print(f"  检查日期: {ds.StudyDate if 'StudyDate' in ds else 'N/A'}")
        print(f"  检查描述: {ds.StudyDescription if 'StudyDescription' in ds else 'N/A'}")
        print(f"  检查ID: {ds.StudyID if 'StudyID' in ds else 'N/A'}\n")

        print(f"序列信息:")
        print(f"  序列描述: {ds.SeriesDescription if 'SeriesDescription' in ds else 'N/A'}")
        print(f"  序列日期: {ds.SeriesDate if 'SeriesDate' in ds else 'N/A'}")
        print(f"  序列编号: {ds.SeriesNumber if 'SeriesNumber' in ds else 'N/A'}\n")

        print(f"图像信息:")
        print(f"  图像尺寸: {ds.Rows} x {ds.Columns}")
        print(f"  像素间距: {ds.PixelSpacing if 'PixelSpacing' in ds else 'N/A'}")
        print(f"  切片厚度: {ds.SliceThickness if 'SliceThickness' in ds else 'N/A'}")
        print(f"  图像位置: {ds.ImagePositionPatient if 'ImagePositionPatient' in ds else 'N/A'}")
        print(f"  图像方向: {ds.ImageOrientationPatient if 'ImageOrientationPatient' in ds else 'N/A'}")
        print(f"  像素表示: {ds.PixelRepresentation if 'PixelRepresentation' in ds else 'N/A'}")
        print(f"  Bits分配: {ds.BitsAllocated if 'BitsAllocated' in ds else 'N/A'}")

        return ds

    except FileNotFoundError:
        print(f"错误: 找不到文件 {dicom_path}")
    except Exception as e:
        print(f"处理DICOM文件时出错: {str(e)}")
    return None


if __name__ == "__main__":
    # 直接使用你的DICOM文件路径
    dicom_file_path = "/home/lyq/Desktop/DWI_HL/YN2_DWI/DWI_high/1392944_800 /1392944_20151118_3_1443.dcm"

    if os.path.exists(dicom_file_path):
        dicom_dataset = print_dicom_info(dicom_file_path)

        # 如果需要查看所有元数据字段，可以取消下面这行的注释
        # print("\n所有DICOM元数据字段:\n", dicom_dataset)
    else:
        print(f"错误: DICOM文件不存在 - {dicom_file_path}")
