import sys
import logging
from Ai.deepseek_ai import deepseek


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

logging.getLogger("httpx").setLevel(logging.WARNING)

def deepseek_main():
        logging.info("CLI已启动")
        logging.info("输入[Quit]或[Exit]结束对话")

        while True:
                try:
                        user_input = input("\n>>>.").strip()
                        if user_input.lower() in ["quit", "exit"]:
                                logging.info("安全退出，本轮任务已结束。")
                                break

                        if not user_input:
                                continue

                        deepseek(user_input) # DeepSeek_API调用

                except KeyboardInterrupt:
                        print("\n")
                        logging.warning("Ctrl+C中断任务，任务已退出。")
                        sys.exit(0)
                except Exception as e:
                        logging.error(f":发生严重错误 {e}", exc_info=True)
                        print("请检查网络或api余额后重试")

if __name__ == '__main__':
        deepseek_main()
