import logging
from typing import List
import re, json, html
from pathlib import Path

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 可维护的敏感词 txt（每行一个词），放在项目根 data/sensitive_words.txt
SENSITIVE_TXT = Path(__file__).parent.parent / "data" / "sensitive_words.txt"

# 默认内置关键词
DEFAULT_SENSITIVE = {"自杀", "暴力", "癌症", "色情", "血腥"}

# 注入关键片段正则：```、system:、assistant:、user:、<script>、@everyone 等
INJ_PATTERN = re.compile(
    r"(?:```|system:|assistant:|user:|<script>|@everyone|@@)",
    re.IGNORECASE,
)

def _load_sensitive() -> set[str]:
    if SENSITIVE_TXT.exists():
        extra = {w.strip() for w in SENSITIVE_TXT.read_text(encoding="utf-8").splitlines() if w.strip()}
        return DEFAULT_SENSITIVE | extra
    return DEFAULT_SENSITIVE

_SENSITIVE_SET = _load_sensitive()


def preprocess_input(text: str) -> str:
    """输入预处理：屏蔽敏感词 + 去除注入片段 + 简单 XSS 转义"""
    # 过滤注入
    text = INJ_PATTERN.sub("", text)

    # 敏感词替换
    for word in _SENSITIVE_SET:
        text = text.replace(word, "***")

    # 防止 HTML 注入
    text = html.escape(text, quote=False)

    # 去掉多余空格
    return re.sub(r"\s{2,}", " ", text).strip()

def sanitize_output(text: str) -> str:
    """
    后置清洗：
      • 去掉注入关键片段
      • 保留 <details>/<summary>，其余做 HTML escape
    """
    # 去掉注入关键词
    cleaned = INJ_PATTERN.sub("", text)

    # 占位符表，同时处理 <summary>
    placeholders = {
        "<details>":  "§details_open§",
        "</details>": "§details_close§",
        "<summary>":  "§summary_open§",
        "</summary>": "§summary_close§",
    }

    # 暂存
    for tag, ph in placeholders.items():
        cleaned = cleaned.replace(tag, ph)

    # HTML 转义（防 XSS）
    cleaned = html.escape(cleaned, quote=False)

    # 还原占位符
    for tag, ph in placeholders.items():
        cleaned = cleaned.replace(ph, tag)

    return cleaned



COLORS = {"低": "🟩", "中": "🟨", "高": "🟥"}

def _join_list(item):
    return ", ".join(item) if isinstance(item, list) else item


def clean_json_text(raw: str) -> str:
    """
    尝试从文本中提取合法 JSON 并修复常见格式问题：
    - 去掉 ```json ... ``` 包裹
    - 截取第一个 { 到最后一个 }
    - 单引号替换成双引号
    - 删除结尾多余逗号
    """
    # 去掉 ```json ... ``` 代码块标记
    raw = raw.replace("```json", "").replace("```", "")
    # 截取花括号
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    # 单引号 → 双引号
    raw = re.sub(r"'", '"', raw)
    # 尾逗号
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)
    return raw.strip()


def format_output(answer: str) -> dict:
    """解析 JSON -> 生成可渲染 Markdown，附 raw"""
    disclaimer = "\n\n※ 本建议仅供参考，不能替代专业医疗诊断。如有紧急情况请立即就医。"
    if not answer or not isinstance(answer, str):
        return {"formatted": "⚠️ 模型未返回内容" + disclaimer, "raw": None}

    safe_json = clean_json_text(answer)

    try:
        data = json.loads(safe_json)
    except Exception:
        # —— 新增：截取第一个 { 到最后一个 } 再尝试一次 ——
        if "{" in safe_json and "}" in safe_json:
            snippet = safe_json[safe_json.find("{"): safe_json.rfind("}") + 1]
            try:
                data = json.loads(re.sub(r"'", '"', snippet))
            except Exception:
                return {"formatted": answer + disclaimer, "raw": None}
        else:
            return {"formatted": answer + disclaimer, "raw": None}


    # 建议 bullet
    sug_raw = data.get("suggestion", "")
    parts = re.split(
        r"[;\n]|(?:\d+\.(?=\s|[（【\(]))|(?:、)",   # ← 括号已配平
        sug_raw
    )
    parts = [p.strip() for p in parts if p.strip()]
    sug_fmt = "\n".join(f"- {p}" for p in parts) if parts else "- （暂无建议）"

    # 折叠问题树
    tree_md = ""
    causes = data.get("possible_causes", [])
    if isinstance(causes, list) and causes:
        if isinstance(causes[0], dict):
            for c in causes:
                tree_md += (
                    f"<details><summary>🔍 {c.get('name')}</summary>\n\n"
                    f"- **怀疑理由**：{c.get('reason', '–')}\n"
                    f"- **优先检查**：{c.get('test', '–')}\n\n"
                    "</details>\n\n"
                )
        else:
            tree_md = _join_list(causes)
    else:
        tree_md = "（暂无可疑病因）"



    risk   = data.get("risk_level", "未知")
    color  = COLORS.get(risk, "⬜️")
    conf   = data.get("confidence", "?")

    formatted = (
        f"🗣️ **直接回应**：{data.get('direct_reply')}\n\n"
        f"{color} **风险等级**：{risk}（置信度 {conf}）\n\n"
        f"📝 **初步诊断**：{data.get('answer')}\n\n"
        f"💡 **进一步建议**：\n{sug_fmt}\n\n"
        f"🏥 **就诊时限**：{data.get('consult_urgency')}  |  **建议科室**：{data.get('recommended_department')}\n\n"
        f"💭 **可能原因**：\n{tree_md}"
    )
    formatted = sanitize_output(formatted)
    return {"formatted": formatted + disclaimer, "raw": answer.strip()}

def extract_direct_reply(raw_json: str) -> str:
    """从 JSON 字符串里取出 direct_reply；解析失败就裁剪前 60 字符"""
    try:
        data = json.loads(clean_json_text(raw_json))
        return data.get("direct_reply") or data.get("answer", "")[:60]
    except Exception:
        return raw_json[:60]


from pydantic import BaseModel, Field, ValidationError
from pydantic.types import confloat

class Cause(BaseModel):
    name: str
    reason: str
    test: str

class AnswerSchema(BaseModel):
    direct_reply: str
    answer: str
    suggestion: str
    risk_level: str
    confidence: confloat(ge=0, le=1)
    consult_urgency: str
    possible_causes: list[Cause] = Field(..., min_length=0)
    recommended_department: str

def validate_json(text: str) -> bool:
    try:
        cleaned = clean_json_text(text)
        logger.info("🧪 validate_json 尝试解析:\n%s", cleaned)
        data = json.loads(cleaned)
        AnswerSchema.model_validate(data)
        return True
    except Exception as e:
        logger.error("❌ validate_json 失败，原因：%s", repr(e))
        return False
