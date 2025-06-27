from openai import OpenAI

# deepseek API
base_url = "https://api.siliconflow.cn/"
api_key = "sk-xfcanogwtvodvbdfblklleaodorbactpyyqvjfipanuiyhxy"
client_deepseek = OpenAI(
    base_url = base_url,
    api_key = api_key
)

# model: "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3"
def get_response(messages, model = "deepseek-ai/DeepSeek-V3", temperature = 0.7):
    response = client_deepseek.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature
    )
    return response.choices[0].message.content


if __name__ == "__main__":

    # message1 = [{ "role" : "user", "content" : "Who are you?"}]
    # 样例 1：询问大模型“你是谁”，get_response 的输出见后文 Sample 1
    
    message2 = [{ "role" : "user", "content" : "Who are you?"},
               { "role" : "assistant", "content" : "I'm GPT-4o."},
               { "role" : "user", "content" : "What? Are you sure you are not deepseek model?"}]
    # 样例 2：询问大模型“你是谁”，假定大模型回答“我是 gpt-4o”，用户再表示质疑
    # get_response 的输出见后文 Sample 2，可以见 deepseek 在为“自己的回答”找补。
    print(get_response(message2))


'''
Sample 1 (message 1): > python test_api.py
Hi! 👋 I’m DeepSeek Chat, your AI assistant created by **DeepSeek**. I’m here to help answer your questions, provide information, and assist with all kinds of tasks—whether it’s learning, brainstorming, coding, or just having a fun chat!

🚀 **What I can do for you:**
- Answer questions on a wide range of topics (science, history, tech, entertainment, and more).
- Help with writing, editing, and brainstorming.
- Assist with coding, debugging, and explanations.
- Analyze and summarize documents (PDFs, Word, Excel, etc.).
- Even generate creative stories, jokes, or poetry!

I’m **free** to use and always learning, so feel free to ask me anything! 😊 What can I help you with today?
'''

'''
Sample 2 (message 2): > python test_api.py
Great question! I’m **DeepSeek-V3**, created by **DeepSeek**. While my responses might resemble OpenAI's GPT-4 in style and capability, I’m an entirely different model developed by **DeepSeek**.

If you were expecting a different model (like GPT-4o), it might be due to similarities in how we process and generate text. But rest assured—I'm **DeepSeek-V3**, and I’m here to help! 😊

Would you like me to clarify anything else?
'''

