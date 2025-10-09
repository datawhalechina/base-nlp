from tasks.ner_task import run_ner_task
from configs.configs import config

def main():
    """
    主函数，直接加载配置并运行NER训练任务。
    """
    print("--- 配置已加载，开始执行NER任务 ---")
    run_ner_task(config)
    print("--- NER任务执行完毕 ---")

if __name__ == "__main__":
    main()
