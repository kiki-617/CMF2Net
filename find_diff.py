import os


def extract_identifier(filename):
    """提取下划线后的第二个元素作为标识符"""
    # 移除扩展名
    for ext in ['.nii.gz', '.nii']:
        if filename.endswith(ext):
            filename = filename[:-len(ext)]
            break

    # 按下划线分割并取第二个元素
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]
    else:
        return filename  # 如果没有下划线，返回整个文件名


def get_identifiers(folder_path):
    """获取文件夹中所有文件的标识符"""
    identifiers = {}
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return identifiers

    for filename in os.listdir(folder_path):
        if filename.endswith(('.nii', '.nii.gz')):
            identifier = extract_identifier(filename)
            if identifier:
                if identifier not in identifiers:
                    identifiers[identifier] = []
                identifiers[identifier].append(filename)
    return identifiers


def find_extra_files(folder1, folder2):
    """查找两个文件夹中交集外的文件"""
    ids1 = get_identifiers(folder1)
    ids2 = get_identifiers(folder2)

    intersection = set(ids1.keys()) & set(ids2.keys())
    only_in_folder1 = set(ids1.keys()) - intersection
    only_in_folder2 = set(ids2.keys()) - intersection

    return {
        'intersection': intersection,
        'only_in_folder1': only_in_folder1,
        'only_in_folder2': only_in_folder2,
        'folder1_files': ids1,
        'folder2_files': ids2
    }


def print_results(results, folder1, folder2):
    """打印对比结果"""
    print(f"文件夹1: {folder1}")
    print(f"文件夹2: {folder2}")
    print("-" * 70)

    print(f"交集标识符数量: {len(results['intersection'])}")
    print(f"仅在文件夹1中的标识符: {len(results['only_in_folder1'])}")
    print(f"仅在文件夹2中的标识符: {len(results['only_in_folder2'])}")
    print("-" * 70)

    # 打印交集文件
    if results['intersection']:
        print("\n交集文件（两边都有的标识符对应的文件）:")
        for identifier in sorted(results['intersection']):
            files1 = results['folder1_files'][identifier]
            files2 = results['folder2_files'][identifier]
            print(f"  标识符: {identifier}")
            print(f"    文件夹1: {', '.join(files1)}")
            print(f"    文件夹2: {', '.join(files2)}")

    # 打印仅在文件夹1中的文件
    if results['only_in_folder1']:
        print("\n仅在文件夹1中存在的文件:")
        for identifier in sorted(results['only_in_folder1']):
            files = results['folder1_files'][identifier]
            print(f"  标识符: {identifier}")
            print(f"    文件: {', '.join(files)}")

    # 打印仅在文件夹2中的文件
    if results['only_in_folder2']:
        print("\n仅在文件夹2中存在的文件:")
        for identifier in sorted(results['only_in_folder2']):
            files = results['folder2_files'][identifier]
            print(f"  标识符: {identifier}")
            print(f"    文件: {', '.join(files)}")


def main():
    # 设置两个文件夹的路径
    folder1 = "/home/lyq/Desktop/中国医科大基线2+3/T2/image_res/"
    folder2 = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/图像配准实验/reference_DCE/CMU/SyNRA/T2/"

    # 查找差异文件
    results = find_extra_files(folder1, folder2)

    # 打印结果
    print_results(results, folder1, folder2)


if __name__ == "__main__":
    main()
