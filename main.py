from modules.retriever import MedicalRetrieverOnline
from modules.retriever import MedicalRetrieverOffline
from modules.generator import DeepSeekGenerator
from modules.utils import preprocess_input, format_output

def main():
    # 初始化组件
    generator = DeepSeekGenerator()
    dialogue_history = []
    print("医疗问答助手已启动，输入'exit'退出对话")
    while True:
        # 用户输入
        user_query = input("\n患者: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        # 预处理查询
        clean_query = preprocess_input(user_query)
        
        try:
            # 生成回答（传入历史对话）
            answer = generator.generate_answer(
                query=clean_query,
                dialogue_history=dialogue_history
            )
            
            if answer:
                formatted_answer = format_output(answer)
                print(f"\n助手: {formatted_answer}")
                # 更新对话历史（保留最近3轮对话）
                dialogue_history.extend([
                    {"role": "user", "content": clean_query},
                    {"role": "assistant", "content": answer}
                ])
                # 保持对话历史不超过6条（3轮）
                dialogue_history = dialogue_history[-6:]
            else:
                print("\n助手: 暂时无法回答这个问题，请尝试更详细的描述")
                
        except Exception as e:
            print(f"\n系统错误: {str(e)}")

if __name__ == "__main__":
    main()