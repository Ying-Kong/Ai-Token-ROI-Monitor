import sys
import logging
from deepseek_ai import deepseek


def deepseek_main():
        try:
                deepseek() # DeepSeek_API调用
                logging.info("本轮任务已结束。")
        except KeyboardInterrupt:
                print("\n")
                logging.warning("中断任务，任务已安全退出。")
                sys.exit(0)
        except Exception as e:
                logging.error(f":发生严重错误 {e}", exc_info=True)
                sys.exit(1)

if __name__ == '__main__':
        deepseek_main()
