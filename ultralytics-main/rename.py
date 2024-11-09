import os


def rename_files_in_folder(folder_path, prefix="train_", extension=""):
    # 获取文件夹中的所有文件
    files = sorted(os.listdir(folder_path))

    # 遍历并重命名每个文件
    for i, filename in enumerate(files, 1):
        # 获取文件的扩展名
        file_extension = os.path.splitext(filename)[1]

        # 如果指定了扩展名过滤器，只处理匹配的文件
        if extension and not filename.endswith(extension):
            continue

        # 生成新的文件名，编号按四位数字格式化
        new_name = f"{prefix}{i:04d}{file_extension}"

        # 获取文件的完整路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")


# 示例：将目标文件夹下的所有文件重命名为 train_0001, train_0002 这样的格式
folder_path = "F:/YOLO/ultralytics-main/datasets/images"  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)
