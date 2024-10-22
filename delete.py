# import os
# import re

# def extract_number(filename):
#     # 使用正则表达式提取文件名中的数字部分
#     match = re.search(r'(\d+)', filename)
#     return int(match.group(1)) if match else float('inf')
# directory_path = './data'
# file_list = sorted([
#     os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
#     ],key=lambda f: extract_number(os.path.basename(f)))

# # 删除前 num_files_to_delete 个文件
# for f in file_list[:10]:
#     try:
#         os.remove(f)
#         print(f"Deleted: {f}")
#     except Exception as e:
#         print(f"Error deleting {f}: {e}")
import os

def delete_corrupted_files(directory_path, size_threshold_mb=5):
    # 将大小阈值转换为字节
    size_threshold_bytes = size_threshold_mb * 1024 * 1024

    # 遍历目录中的所有文件
    for f in os.listdir(directory_path):
        file_path = os.path.join(directory_path, f)

        if os.path.isfile(file_path):  # 确保是文件
            file_size = os.path.getsize(file_path)  # 获取文件大小
            
            # 检查文件大小是否小于阈值
            if file_size < size_threshold_bytes:
                try:
                    os.remove(file_path)  # 删除文件
                    print(f"Deleted: {file_path}, Size: {file_size} bytes")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# 示例调用
directory_path = './data'  # 替换为你的目录路径
delete_corrupted_files(directory_path)
