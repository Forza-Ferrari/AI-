from baidusearch.baidusearch import search
import re
from langchain.schema import Document
class BaiduSearcher:
    @staticmethod
    def _get_fallback_knowledge(self, query):
        """从公共医学知识库获取数据（示例）"""
        fallback_sources = [
            {"title": "默沙东诊疗手册", "url": "https://www.msdmanuals.com"},
            {"title": "UpToDate临床顾问", "url": "https://www.uptodate.com"}
        ]
        return [Document(
            page_content=f"【公共知识】{query}的通用医学建议",
            metadata={"source": src["url"], "fallback": True}
        ) for src in fallback_sources]

    def search_medical_info(self, query, top_k=3):
         # 扩大搜索基数并添加医学关键词增强
        results = []
        query = f"{query} 医学"
        results.append(search(query, num_results=top_k))
        
        filtered = []
        for res in results[0]:
            abst = res['abstract'].replace("\n", "")
            filtered.append(Document(
                page_content=f"标题：{res['title']}\n摘要：{abst}",
                metadata={"source": res["url"]}
            ))
        
        # 兜底逻辑：添加公共知识库
        if not filtered:
            public_knowledge = self._get_fallback_knowledge(query)
            filtered.extend(public_knowledge)
        return filtered
    
    