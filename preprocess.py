import cv2
import os
import numpy as np

def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")

def process_single_image(img_path, output_path, brightness=30):
    """
    处理单张图片的流水线：读取 -> 灰度化 -> 亮度调节 -> 保存
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # 1. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 亮度调节 (beta 为正变亮，为负变暗)
    processed = cv2.convertScaleAbs(gray, alpha=1.0, beta=brightness)
    
    # 3. 保存图片
    cv2.imwrite(output_path, processed)
    return True

def batch_process(input_dir, output_dir, brightness=30):
    """
    批量处理文件夹下的所有图片
    """
    ensure_dir(output_dir)
    all_files = os.listdir(input_dir)
    count = 0
    
    print(f"开始处理文件夹: {input_dir}")
    for file_name in all_files:
        # 过滤图片格式
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"proc_{file_name}")
            
            if process_single_image(input_path, output_path, brightness):
                print(f"成功: {file_name}")
                count += 1
            else:
                print(f"失败: {file_name} (无法读取)")
    
    print(f"--- 处理完成，共计 {count} 张图片已存入 {output_dir} ---")

if __name__ == "__main__":
    # 这个部分仅当你直接运行 python preprocess.py 时执行
    # 方便你单独测试这个文件
    batch_process('data', 'processed_data')