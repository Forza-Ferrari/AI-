# modules/generator.py

from typing import Optional
from pathlib import Path
import json
from openai import OpenAI
from config import settings
from modules.utils import logger, validate_json
from modules.retriever import MedicalRetrieverOnline
from modules.retriever import MedicalRetrieverOffline
from typing import Optional, List, Dict

client_deepseek = OpenAI(
    base_url=settings.DEEPSEEK_BASE_URL,
    api_key=settings.DEEPSEEK_API_KEY
)

def get_response(messages, model="deepseek-ai/DeepSeek-V3", temperature=0.7):
    resp = client_deepseek.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return resp.choices[0].message.content

class DeepSeekGenerator:
    @staticmethod
    def generate_answer(
        query: str,
        dialogue_history: List[Dict], 
        top_k: int = 50,
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.7
    ) -> Optional[str]:
        try:
            fallback_json = {
                "answer": "抱歉，我暂时无法提供有效建议。",
                "suggestion": "请尝试描述得更详细一些，或者稍后再试。",
                "risk_level": "未知",
                "possible_causes": [],
                "recommended_department": "无"
            }
            # 1. 用检索器挑出 top_k 条最相关的 contexts
            retriever1 = MedicalRetrieverOnline()
            retriever2 = MedicalRetrieverOffline()
            docs1 = retriever1.hybrid_retrieve(query, top_k=top_k)
            docs2 = retriever2.hybrid_retrieve(query, top_k=top_k)
            contexts1 = [doc.page_content for doc in docs1]
            contexts2 = [doc.page_content for doc in docs2]
            # contexts2 = []
            # 2. 构建带历史上下文的prompt
            joined_contexts1 = "\n".join(f"- {ctx}" for ctx in contexts1)
            joined_contexts2 = "\n".join(f"- {ctx}" for ctx in contexts2)
            system_content = (
                "你是一名专业医疗助理，正在与用户进行多轮对话。\n"
                "请结合以下医学资料和历史对话认真回答用户问题。\n\n"
                "【网页搜索信息】\n"
                f"{joined_contexts1}\n\n"
                "【权威手册信息】\n"
                f"{joined_contexts2}\n\n"
                "请以以下 JSON 输出，不得添加 JSON 外文字：\n"
                "{\n"
                '  "direct_reply": "",\n'
                '  "answer": "",\n'
                '  "suggestion": "",\n'
                '  "risk_level": "低 / 中 / 高",\n'
                '  "confidence": 0.0,\n'
                '  "consult_urgency": "立即就医 / 48h 内 / 观察即可",\n'
                '  "possible_causes": ["..."],\n'
                '  "recommended_department": "",\n'
                "}\n"
                "⚠️ possible_causes 和 references 必须是 **双引号** 包裹的 JSON 数组。"
            )
            
            system_content += (
                "\n\n请将 possible_causes 字段扩展为对象数组，每个元素包含：\n"
                '  • "name"：疾病名称\n'
                '  • "reason"：为什么怀疑它（典型症状/流行病学）\n'
                '  • "test"：首选排查方式(影像/实验室/体格)\n'
                "示例：\n"
                '"possible_causes": [\n'
                '  {"name":"肺炎","reason":"低烧+咳嗽+湿啰音","test":"胸片"},\n'
                '  {"name":"肺结核","reason":"低热盗汗体重减轻","test":"胸片/痰涂片"}\n'
                "]\n"
                "请确保返回内容严格为 JSON 格式，开头必须是 {，不得包含多余句子或标点。"
            )



            # 3. 构建消息列表（包含系统提示、历史对话和当前问题）
            messages = [{"role": "system", "content": system_content}]
            
            # 添加历史对话记录
            for entry in dialogue_history:
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })
            
            # 添加当前问题
            messages.append({"role": "user", "content": query})

            # 4. 调用生成接口 —— 加重试
            attempt = 0
            temp = temperature
            while attempt < 3:
                result = get_response(messages, model=model, temperature=temp)
                logger.info("🟢 原始模型输出:\n%s", result)
                if validate_json(result):
                    return result
                attempt += 1
                temp = max(0.5, temp - 0.2)

            # 三次都失败
            return json.dumps(fallback_json)

            

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return json.dumps(fallback_json)
        
    @staticmethod
    def stream_answer(
        messages,
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.7,
    ):
        """
        按 token 流式产出，用于打字机效果。
        调用方式与 generate_answer 保持同一 messages 结构。
        """
        resp = client_deepseek.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,          # 关键参数
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    @staticmethod
    def stream_natural_reply(
        query: str,
        dialogue_history,
        model: str = "deepseek-ai/DeepSeek-V3",
        temperature: float = 0.7,
    ):
        """
        与 stream_generate_answer 相同检索，但要求模型只用自然语言回复。
        """
        # --- 检索与 system_content 与 stream_generate_answer 相同 ---
        retriever1 = MedicalRetrieverOnline()
        retriever2 = MedicalRetrieverOffline()
        ctx1 = "\n".join(f"- {d.page_content}"
                         for d in retriever1.hybrid_retrieve(query, top_k=20))
        ctx2 = "\n".join(f"- {d.page_content}"
                         for d in retriever2.hybrid_retrieve(query, top_k=20))

        system_content = (
            "你是一名专业医疗助理，正在进行多轮对话。\n"
            "以下信息供参考（请勿直接引用原文）：\n"
            f"{ctx1}\n{ctx2}\n"
            "请**仅用中文自然语言**回答，不要使用任何花括号、引号或 JSON 格式标记。"
        )

        messages = [{"role": "system", "content": system_content}]
        messages += dialogue_history
        messages.append({"role": "user", "content": query})

        resp = client_deepseek.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


