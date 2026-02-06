from preprocess import batch_process
# 定义你的路径
INPUT_PATH = 'data'
OUTPUT_PATH = 'processed_data'

def main():
    print("--- CV 项目启动 ---")
    
    # 直接调用预处理模块
    batch_process(INPUT_PATH, OUTPUT_PATH, brightness=40)
    
    # 以后在这里添加模型训练的代码
    # train_model()
    
    print("--- 任务结束 ---")

if __name__ == "__main__":
    main()