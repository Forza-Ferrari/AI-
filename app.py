import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARN"] = "true"
os.environ["STREAMLIT_WATCHDOG"] = "false"
import streamlit as st
from modules.generator import DeepSeekGenerator
from modules.utils import preprocess_input, format_output
from modules.utils import extract_direct_reply 

# ---------------- 页面基设 ----------------
st.set_page_config(page_title="医疗智能问答助手", layout="wide")
st.title("🩺 医疗智能问答助手")
st.markdown("请输入症状描述，AI 将结合医学资料与对话历史为您提供建议。")

# ---------------- 会话状态 ----------------
if "history" not in st.session_state:
    st.session_state.history = []

generator = DeepSeekGenerator()

# ---------------- 工具函数 ----------------
def stream_and_render(user_input: str):
    """流式回答 -> 结束后结构化解析"""
    clean_q = preprocess_input(user_input)

    # 构造 messages（含历史）
    messages = [{"role": "system", "content": "你是一名医疗助理"}]
    for h in st.session_state.history:
        messages.append(h)
    messages.append({"role": "user", "content": clean_q})

    placeholder = st.empty()
    full_text = ""

    # 1️⃣ 打字机效果
    for tok in generator.stream_answer(messages, temperature=0.7):
        full_text += tok
        placeholder.markdown(full_text + "▌")

    # 2️⃣ 结构化解析
    result = format_output(full_text)
    placeholder.markdown(result["formatted"], unsafe_allow_html=True)

    if result.get("raw"):
        with st.expander("🗒 查看原始 JSON"):
            st.code(result["raw"], language="json")

    # 3️⃣ 更新历史（保留 3 轮）
    st.session_state.history.extend([
        {"role": "user", "content": clean_q},
        {"role": "assistant", "content": full_text}
    ])
    st.session_state.history = st.session_state.history[-6:]

def nonstream_answer(user_input: str, temp: float):
    """一次性回答，用于参数对比。返回 plain + formatted"""
    raw_json = generator.generate_answer(
        query=user_input,
        dialogue_history=st.session_state.history,
        temperature=temp
    )
    formatted = format_output(raw_json)
    plain     = extract_direct_reply(raw_json)
    # 只返回用得到的两项
    return {"plain": plain, "formatted": formatted["formatted"]}


def stream_and_replace(user_input: str):
    clean_q = preprocess_input(user_input)
    hist = st.session_state.history

    # ---- 1️⃣ 自然语言流式 ----
    placeholder = st.empty()
    nat_text = ""
    for tok in generator.stream_natural_reply(
        query=clean_q,
        dialogue_history=hist,
        temperature=0.7,
    ):
        nat_text += tok
        placeholder.markdown(nat_text + "▌")

    # ---- 2️⃣ 结构化二次调用（非流式）----
    json_answer = generator.generate_answer(
        query=clean_q,
        dialogue_history=hist,
        temperature=0.7
    )
    result = format_output(json_answer)

    # 替换 placeholder 内容
    placeholder.markdown(result["formatted"], unsafe_allow_html=True)

    if result["raw"]:
        with st.expander("🗒 查看原始 JSON"):
            st.code(result["raw"], language="json")

    # 更新历史
    # 去掉光标 ▌ 与多余空白
    nat_clean = nat_text.rstrip("▌").strip()

    st.session_state.history.extend([
        {"role": "user", "content": clean_q},
        {
            "role": "assistant",
            "content": nat_clean,            # 保存真实流式自然语言
            "formatted": result["formatted"] # 同时保存结构化卡片
        }
    ])
    st.session_state.history = st.session_state.history[-6:]



# ---------------- 输入区 ----------------
user_input = st.text_area(
    "💬 请输入您的症状描述：",
    placeholder="例如：最近持续低烧，咳嗽加重，有必要就医吗？",
    height=150
)

# ---------------- 按钮区 ----------------
col1, col2 = st.columns(2)
with col1:
    stream_btn  = st.button("🎯 实时回答（流式）")
with col2:
    compare_btn = st.button("🔬 参数对比 (0.7 vs 1.2)")

# ---------------- 事件处理 ----------------
if stream_btn and user_input.strip():
    with st.spinner("AI 正在回答..."):
        stream_and_replace(user_input)

if compare_btn and user_input.strip():
    with st.spinner("AI 正在生成对比..."):
        low  = nonstream_answer(user_input, 0.7)
        high = nonstream_answer(user_input, 1.2)

    st.markdown("### 🔍 参数对比 (temperature 0.7 vs 1.2)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🧊 0.7")
        st.markdown(low["formatted"], unsafe_allow_html=True)
    with c2:
        st.subheader("🔥 1.2")
        st.markdown(high["formatted"], unsafe_allow_html=True)

# ---------------- 历史记录展示 ----------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 对话历史（最近 3 轮）")
    for i in range(0, len(st.session_state.history), 2):
        user_msg = st.session_state.history[i]["content"]
        ai_item  = st.session_state.history[i + 1]

        st.markdown(f"👤 **用户：** {user_msg}")

        if "formatted" in ai_item:
            with st.expander("🤖 **助手（自然语言）**"):
                st.write(ai_item["content"])
            st.markdown(ai_item["formatted"], unsafe_allow_html=True)
        else:
            st.markdown(f"🤖 **助手：** {ai_item['content']}")

# ---------------- 清空按钮 ----------------
if st.button("🧹 清除对话历史"):
    st.session_state.history = []
    st.experimental_rerun()

# ---------------- 免责声明 ----------------
st.markdown("---")
st.caption("⚠️ 本建议仅供参考，不能替代专业医疗诊断。如有严重不适，请及时就医。")
