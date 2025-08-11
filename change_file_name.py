import os


def rename_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii.gz'):
                if '_DCE_' in file:
                    new_name = file.replace('_DCE_', '_')
                elif '_T2_' in file:
                    new_name = file.replace('_T2_', '_')
                else:
                    continue
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_name)
                os.rename(old_file_path, new_file_path)
                print(f"已将 {old_file_path} 重命名为 {new_file_path}")


# 请替换为你的训练、验证和测试文件夹路径
train_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/images'
val_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/imagesVal'
test_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/imagesTs'
# 请替换为你的训练、验证和测试文件夹路径
train_label_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/labels'
val_label_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/labelsTs'
test_label_folder = '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/nnUNet_raw_data/Task131_DCET2Reg/labelsVal'

# 重命名训练文件夹中的文件
rename_files(train_folder)
# 重命名验证文件夹中的文件
rename_files(val_folder)
# 重命名测试文件夹中的文件
rename_files(test_folder)

rename_files(train_label_folder)
rename_files(val_label_folder)
rename_files(test_label_folder)