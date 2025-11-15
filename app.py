import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os
import time
import logging
import sys
import uuid
from typing import Tuple, Optional
from dotenv import load_dotenv

# ========================
# 环境配置
# ========================
load_dotenv()  # 加载环境变量

# 从环境变量读取配置
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-zh")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_knowledge_base")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder:480b-cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"CHROMA_DB_DIR = {CHROMA_DB_DIR}")
print(f"EMBEDDINGS_MODEL = {EMBEDDINGS_MODEL}")
print(f"COLLECTION_NAME = {COLLECTION_NAME}")
print(f"OLLAMA_MODEL = {OLLAMA_MODEL}")
print(f"OLLAMA_BASE_URL = {OLLAMA_BASE_URL}")

# 应用配置
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "1800"))  # 30分钟

# ========================
# 日志配置
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ========================
# 主题配置
# ========================
THEMES = {
    "DeepSeek 蓝": {
        "primary": "#1e40af",
        "secondary": "#3b82f6",
        "accent": "#60a5fa",
        "bg": "#f8fafc",
        "card_bg": "#ffffff",
        "text": "#1e293b",
        "border": "#e2e8f0"
    },
    "深色模式": {
        "primary": "#1e40af",
        "secondary": "#3b82f6",
        "accent": "#60a5fa",
        "bg": "#0f172a",
        "card_bg": "#1e293b",
        "text": "#f1f5f9",
        "border": "#334155"
    },
    "绿色科技": {
        "primary": "#059669",
        "secondary": "#10b981",
        "accent": "#34d399",
        "bg": "#f0fdf4",
        "card_bg": "#ffffff",
        "text": "#064e3b",
        "border": "#a7f3d0"
    },
    "紫色梦幻": {
        "primary": "#7c3aed",
        "secondary": "#8b5cf6",
        "accent": "#a78bfa",
        "bg": "#faf5ff",
        "card_bg": "#ffffff",
        "text": "#4c1d95",
        "border": "#c4b5fd"
    }
}

# ========================
# 页面配置
# ========================
st.set_page_config(
    page_title="智能知识库问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ========================
# 会话状态初始化
# ========================
def init_session_state():
    """初始化会话状态"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = {"question": "", "answer": "", "sources": []}

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "theme" not in st.session_state:
        st.session_state.theme = "DeepSeek 蓝"

    if "input_height" not in st.session_state:
        st.session_state.input_height = 120

    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = time.time()

    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    if "initialized" not in st.session_state:
        st.session_state.initialized = False


# ========================
# 应用样式
# ========================
def apply_theme(theme_name):
    """应用选定的主题"""
    theme = THEMES[theme_name]

    css = f"""
    <style>
        /* 主题变量 */
        :root {{
            --primary-color: {theme['primary']};
            --secondary-color: {theme['secondary']};
            --accent-color: {theme['accent']};
            --bg-color: {theme['bg']};
            --card-bg: {theme['card_bg']};
            --text-color: {theme['text']};
            --border-color: {theme['border']};
        }}

        /* 全局样式 */
        .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
        }}

        .main-header {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .chat-container {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
        }}

        .user-message {{
            background: linear-gradient(135deg, {theme['secondary']} 0%, {theme['primary']} 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
        }}

        .assistant-message {{
            background: var(--bg-color);
            color: var(--text-color);
            padding: 1rem 1.5rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.5rem 0;
            max-width: 80%;
            border: 1px solid var(--border-color);
        }}

        .source-files {{
            background: {theme['bg']};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            border-left: 4px solid {theme['accent']};
            color: var(--text-color);
        }}

        .history-item {{
            padding: 0.75rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: var(--text-color);
        }}

        .history-item:hover {{
            background: {theme['accent']};
            color: white;
            transform: translateX(4px);
        }}

        .stButton button {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px {theme['primary']}30;
        }}

        /* 表单按钮特殊样式 */
        .stForm button {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }}

        .stForm button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px {theme['primary']}30 !important;
        }}

        .sidebar-header {{
            background: linear-gradient(135deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }}

        /* 自定义文本区域样式 */
        .stTextArea textarea {{
            background: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-size: 16px;
            min-height: {st.session_state.input_height}px;
        }}

        .stTextArea label {{
            color: var(--text-color);
            font-weight: 600;
        }}

        /* 自定义选择框样式 */
        .stSelectbox div[data-baseweb="select"] {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
        }}

        .stSelectbox label {{
            color: var(--text-color);
        }}

        /* 自定义滑块样式 */
        .stSlider div[data-baseweb="slider"] {{
            color: {theme['primary']};
        }}

        /* 自定义展开器样式 */
        .streamlit-expanderHeader {{
            background: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }}

        .streamlit-expanderContent {{
            background: var(--card-bg);
            color: var(--text-color);
        }}

        .st-emotion-cache-zy6yx3 {{
            padding-top: 3rem !important;
        }}

        /* 隐藏表单边框 */
        form {{
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }}

        .stForm {{
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        /* 表单容器样式 */
        .form-container {{
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }}

        /* 状态指示器 */
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .status-online {{
            background-color: #10b981;
        }}

        .status-offline {{
            background-color: #ef4444;
        }}
        
        .st-emotion-cache-scp8yw{{
            display: none;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ========================
# 工具函数
# ========================
def get_file_name(doc):
    """从文档元数据中提取文件名"""
    if "file_name" in doc.metadata:
        return doc.metadata["file_name"]
    elif "source" in doc.metadata:
        return os.path.basename(doc.metadata["source"])
    else:
        return "未知文件"


def check_session_timeout():
    """检查会话是否超时"""
    session_duration = time.time() - st.session_state.session_start_time
    if session_duration > SESSION_TIMEOUT:
        st.session_state.chat_history = []
        st.session_state.current_chat = {"question": "", "answer": "", "sources": []}
        st.session_state.session_start_time = time.time()
        logger.info(f"会话超时重置 - 用户: {st.session_state.user_id}")
        return True
    return False


def rate_limit_check():
    """简单的请求限流"""
    current_time = time.time()
    if current_time - st.session_state.last_request_time < 1:  # 1秒内只能请求一次
        return False
    st.session_state.last_request_time = current_time
    return True


def health_check():
    """系统健康检查"""
    try:
        status = {
            "vectorstore": st.session_state.vectorstore is not None,
            "llm": st.session_state.llm is not None,
            "chroma_db_exists": os.path.exists(CHROMA_DB_DIR),
            "session_duration": time.time() - st.session_state.session_start_time,
            "total_chats": len(st.session_state.chat_history)
        }

        # 测试向量数据库连接
        if status["vectorstore"]:
            try:
                count = st.session_state.vectorstore._collection.count()
                status["document_count"] = count
            except Exception as e:
                status["document_count"] = f"错误: {str(e)}"
        else:
            status["document_count"] = "未初始化"

        return status
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {"error": str(e)}


# ========================
# 初始化向量数据库和 LLM
# ========================
@st.cache_resource(ttl=3600)  # 缓存1小时
def init_components() -> Tuple[Optional[Chroma], Optional[OllamaLLM]]:
    """初始化向量数据库和 LLM，带有完整的错误处理"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"初始化组件，尝试 {attempt + 1}/{MAX_RETRIES}")

            # 检查向量数据库目录是否存在
            if not os.path.exists(CHROMA_DB_DIR):
                logger.error(f"向量数据库目录不存在: {CHROMA_DB_DIR}")
                st.error(f"❌ 向量数据库目录不存在: {CHROMA_DB_DIR}")
                return None, None

            # 初始化 embeddings
            logger.info(f"初始化嵌入模型: {EMBEDDINGS_MODEL}")
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDINGS_MODEL,
                model_kwargs={'device': 'cpu'}
            )

            # 初始化向量数据库
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )

            # 测试向量数据库连接
            test_count = vectorstore._collection.count()
            logger.info(f"向量数据库连接成功，文档数量: {test_count}")

            # 初始化 LLM
            logger.info(f"初始化 LLM: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
            llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                timeout=REQUEST_TIMEOUT
            )

            # 测试 LLM 连接
            test_response = llm.invoke("测试连接，请回复'连接成功'")
            if "连接成功" in test_response or len(test_response) > 0:
                logger.info("LLM 连接测试成功")
            else:
                logger.warning("LLM 连接测试返回异常响应")

            logger.info("所有组件初始化成功")
            return vectorstore, llm

        except Exception as e:
            logger.error(f"初始化失败 (尝试 {attempt + 1}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                st.error(f"❌ 系统初始化失败: {str(e)}")
                return None, None
            time.sleep(2)  # 等待后重试

    return None, None


# ========================
# RAG 问答函数
# ========================
def ask_with_knowledge(query: str):
    """使用知识库回答问题"""
    if not st.session_state.vectorstore:
        return "向量数据库未初始化", []

    if not rate_limit_check():
        return "请求过于频繁，请稍后再试", []

    try:
        # 检索相关文档
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        if not docs:
            return "未找到相关信息。", []

        # 收集文件信息
        file_info = []
        for doc in docs:
            file_name = get_file_name(doc)
            file_type = doc.metadata.get("type", "未知类型")
            source_info = f"{file_name} ({file_type})"
            file_info.append(source_info)

        # 去重文件列表
        unique_files = list(set(file_info))

        # 构建上下文
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
你是一个专业的中文知识问答助手。
请根据以下知识库内容回答用户的问题。
如果知识库中没有相关信息，请如实说明。

知识库内容：
{context}

用户问题：
{query}

请提供准确、有用的回答：
"""

        # 显示加载动画并获取回答
        with st.spinner("🤔 正在思考中..."):
            answer = st.session_state.llm.invoke(prompt)

        logger.info(f"问题回答成功 - 用户: {st.session_state.user_id}, 问题长度: {len(query)}")
        return answer, unique_files

    except Exception as e:
        error_msg = f"系统错误：{str(e)}"
        logger.error(f"问答失败 - 用户: {st.session_state.user_id}, 错误: {str(e)}")
        return error_msg, []


# ========================
# 侧边栏 - 历史记录和设置
# ========================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # 侧边栏头部
        # st.markdown('<div class="sidebar-header"><h3>💬 对话历史</h3></div>', unsafe_allow_html=True)
        # st.markdown('', unsafe_allow_html=True)

        # 清空历史按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空历史", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_chat = {"question": "", "answer": "", "sources": []}
                logger.info(f"历史记录已清空 - 用户: {st.session_state.user_id}")
                st.rerun()

        with col2:
            if st.button("🔄 刷新会话", use_container_width=True):
                st.session_state.current_chat = {"question": "", "answer": "", "sources": []}
                st.rerun()

        st.markdown("---")

        # 显示历史记录
        if not st.session_state.chat_history:
            st.info("暂无历史对话")
        else:
            # 显示最近的历史记录（最新的在最上面）
            for i, chat in enumerate(reversed(st.session_state.chat_history[-20:])):
                question_preview = chat["question"][:50] + "..." if len(chat["question"]) > 50 else chat["question"]
                timestamp = chat.get("timestamp", "")

                if st.button(
                        f"**Q:** {question_preview}\n\n*{timestamp}*",
                        key=f"history_{i}",
                        use_container_width=True
                ):
                    st.session_state.current_chat = chat.copy()
                    st.rerun()

        st.markdown("---")

        # 主题设置
        st.markdown("### 🎨 主题设置")
        theme_options = list(THEMES.keys())
        selected_theme = st.selectbox(
            "选择主题",
            theme_options,
            index=theme_options.index(st.session_state.theme),
            key="theme_selector"
        )

        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()

        # 输入框高度设置
        st.markdown("### 📝 输入框设置")
        input_height = st.slider(
            "输入框高度 (像素)",
            min_value=80,
            max_value=300,
            value=st.session_state.input_height,
            step=20,
            key="input_height_slider"
        )

        if input_height != st.session_state.input_height:
            st.session_state.input_height = input_height
            st.rerun()

        st.markdown("---")

        # 系统状态
        st.markdown("### 🔧 系统状态")
        health_status = health_check()

        if health_status.get("error"):
            st.error("❌ 状态检查失败")
        elif all([health_status["vectorstore"], health_status["llm"], health_status["chroma_db_exists"]]):
            st.success("✅ 系统就绪")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("文档数量", health_status.get("document_count", "未知"))
            with col2:
                st.metric("对话次数", health_status["total_chats"])

            # 会话信息
            session_minutes = int(health_status['session_duration'] // 60)
            st.info(f"🕐 当前会话: {session_minutes} 分钟")
        else:
            st.error("❌ 系统异常")
            if not health_status["chroma_db_exists"]:
                st.error("向量数据库目录不存在")
            if not health_status["vectorstore"]:
                st.error("向量数据库未初始化")
            if not health_status["llm"]:
                st.error("语言模型未连接")

        st.markdown("---")

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            - 💭 **输入问题**：在下方输入框输入您的问题
            - 📚 **知识检索**：系统会自动从知识库中检索相关信息
            - 💬 **历史记录**：左侧可以查看和切换历史对话
            - 🎨 **个性化**：可以切换主题和调整输入框大小
            - ⚡ **性能优化**：系统会自动处理超时和限流

            **支持的文件类型**：
            - 📄 PDF 文档
            - 📝 Word 文档
            - 📊 Excel 表格
            - 🗒️ 文本文件
            - 🖼️ 图片文件（需 OCR 支持）
            """)

        # 调试信息（仅在开发模式显示）
        if os.getenv("DEBUG", "False").lower() == "true":
            with st.expander("🔍 调试信息"):
                st.write(f"用户ID: {st.session_state.user_id}")
                st.write(f"会话开始: {time.ctime(st.session_state.session_start_time)}")
                st.write(f"环境: {os.getenv('ENVIRONMENT', 'development')}")


# ========================
# 主界面
# ========================
def render_main():
    """渲染主界面"""
    # 应用主题
    apply_theme(st.session_state.theme)

    # 检查会话超时
    if check_session_timeout():
        st.warning("⚠️ 会话已超时，历史记录已自动清空")

    # 页面头部
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 智能知识库问答系统</h1>
        <p>基于 RAG 技术的企业级智能问答助手 | 当前主题: {st.session_state.theme}</p>
    </div>
    """, unsafe_allow_html=True)

    # 初始化组件
    if not st.session_state.initialized:
        with st.spinner("🔄 正在初始化系统组件..."):
            st.session_state.vectorstore, st.session_state.llm = init_components()
            st.session_state.initialized = True

    # 显示系统状态提示
    if not st.session_state.vectorstore or not st.session_state.llm:
        st.error("""
        ⚠️ 系统组件初始化失败，请检查：
        - 向量数据库目录是否存在
        - Ollama 服务是否运行
        - 网络连接是否正常

        请联系系统管理员。
        """)
        return

    # 聊天容器
    if st.session_state.current_chat["question"]:
        # 用户问题
        st.markdown(f'<div class="user-message"><strong>您:</strong> {st.session_state.current_chat["question"]}</div>',
                    unsafe_allow_html=True)

        # 助手回答
        if st.session_state.current_chat["answer"]:
            st.markdown(
                f'<div class="assistant-message"><strong>助手:</strong> {st.session_state.current_chat["answer"]}</div>',
                unsafe_allow_html=True)

            # 显示来源文件
            if st.session_state.current_chat["sources"]:
                sources_text = "<br/>".join([f"• {source}" for source in st.session_state.current_chat["sources"]])
                st.markdown(f"""
                <div class="source-files">
                    <strong>📁 参考来源:</strong><br>
                    {sources_text}
                </div>
                """, unsafe_allow_html=True)

    # 输入区域
    st.markdown("### 💭 请输入您的问题：")

    # 使用表单包装输入区域
    with st.form(key="question_form", clear_on_submit=True, border=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)

        question = st.text_area(
            "问题输入框",
            placeholder="例如：请总结一下项目文档的主要内容...\n\n或者详细描述某个具体的技术问题...",
            height=st.session_state.input_height,
            label_visibility="collapsed",
            key="question_input"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            submit_btn = st.form_submit_button("🚀 发送问题", use_container_width=True)
        with col2:
            clear_btn = st.form_submit_button("🗑️ 清空输入", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 处理清空输入按钮
    if clear_btn:
        st.rerun()

    # 处理用户输入
    if submit_btn and question.strip():
        # 添加时间戳
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # 保存当前问题
        st.session_state.current_chat = {
            "question": question,
            "answer": "",
            "sources": [],
            "timestamp": current_time
        }

        # 获取回答
        answer, sources = ask_with_knowledge(question)

        # 更新当前对话
        st.session_state.current_chat["answer"] = answer
        st.session_state.current_chat["sources"] = sources

        # 添加到历史记录
        if not any(chat["question"] == question for chat in st.session_state.chat_history):
            st.session_state.chat_history.append(st.session_state.current_chat.copy())
            logger.info(f"新对话已保存 - 用户: {st.session_state.user_id}, 问题: {question[:50]}...")

        st.rerun()

    # 底部信息
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**🔍 检索设置**")
        st.write("Top-K: 3个相关片段")
        st.write(f"重试次数: {MAX_RETRIES}")

    with col2:
        st.markdown("**🤖 模型信息**")
        st.write(f"嵌入模型: {EMBEDDINGS_MODEL.split('/')[-1]}")
        st.write(f"LLM: {OLLAMA_MODEL}")

    with col3:
        st.markdown("**📚 知识库**")
        if st.session_state.vectorstore:
            try:
                count = st.session_state.vectorstore._collection.count()
                st.write(f"文档片段: {count}")
                st.write(f"存储路径: {CHROMA_DB_DIR}")
            except Exception as e:
                st.write(f"文档片段: 错误")
                logger.error(f"获取文档数量失败: {str(e)}")

    with col4:
        st.markdown("**⚙️ 系统配置**")
        st.write(f"会话超时: {SESSION_TIMEOUT // 60}分钟")
        st.write(f"请求超时: {REQUEST_TIMEOUT}秒")


# ========================
# 运行应用
# ========================
def main():
    """主函数"""
    try:
        # 记录应用启动
        logger.info(f"应用启动 - 环境: {os.getenv('ENVIRONMENT', 'development')}")

        init_session_state()
        render_main()
        render_sidebar()

    except Exception as e:
        logger.error(f"应用运行错误: {str(e)}")
        st.error("""
        🚨 系统发生严重错误

        请尝试以下操作：
        1. 刷新页面
        2. 检查系统日志
        3. 联系技术支持

        错误信息已记录到日志文件。
        """)


if __name__ == "__main__":
    main()
