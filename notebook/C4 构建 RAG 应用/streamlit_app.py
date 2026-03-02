# app.py
import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
import sys 
sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中 
sys.path.append("notebook/C4 构建RAG应用") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from zhipuai_llm import ZhipuaiLLM
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
# 文件/路径定义
PERSIST_DIR = "data_base/vector_db2/chroma"
DOCS_DIR = "data_base/docs"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)


# 从 secrets 中读取 API key（请在 Streamlit Cloud 的 Secrets 中设置）
if "ZHIPUAI_API_KEY" in st.secrets:
    os.environ['ZHIPUAI_API_KEY'] = st.secrets["ZHIPUAI_API_KEY"]
else:
    # 本地测试时可以临时设置环境变量，但不要把密钥写死到代码里
    if "ZHIPUAI_API_KEY" in os.environ:
        pass
    else:
        st.warning("未找到 ZHIPUAI_API_KEY， 请在 Streamlit Secrets 中添加。")

def github_put_file(token: str, repo_full: str, path_in_repo: str, file_bytes: bytes,
                    commit_message: str, branch: str = "main", timeout=30):
    """
    Create or update a file in a GitHub repo using Contents API.
    使用 requests.Session + Retry，带超时和详细错误返回。
    返回 (ok: bool, info: dict_or_str)
    """
    if not token:
        return False, "missing github token"

    # 基本大小限制（避免一口气上传超大文件导致连接被重置）
    MAX_BYTES = 8 * 1024 * 1024  # 8 MB，按需调整或提升（GitHub contents API 对大文件不友好）
    if len(file_bytes) > MAX_BYTES:
        return False, f"file too large ({len(file_bytes)} bytes). Use git push or LFS for large files."

    url = f"https://api.github.com/repos/{repo_full}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "streamlit-app"
    }

    # 配置 session + retry（重试 3 次，针对 idempotent 的 GET/PUT 可用）
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(['GET','PUT','POST','DELETE','HEAD','OPTIONS']))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        # 先检查文件是否存在（获取 sha）
        r = session.get(url, headers=headers, params={"ref": branch}, timeout=timeout)
    except requests.exceptions.SSLError as e:
        return False, f"ssl error when checking existence: {e}"
    except requests.exceptions.ReadTimeout:
        return False, "timeout when checking file existence"
    except requests.exceptions.ConnectionError as e:
        return False, f"connection error when checking existence: {e}"
    except Exception as e:
        return False, f"unexpected error when checking existence: {e}"

    # 准备 payload
    import base64
    b64 = base64.b64encode(file_bytes).decode()
    if r.status_code == 200:
        try:
            sha = r.json().get("sha")
        except Exception:
            sha = None
        payload = {"message": commit_message, "content": b64, "branch": branch, "sha": sha}
    elif r.status_code == 404:
        payload = {"message": commit_message, "content": b64, "branch": branch}
    else:
        # 把响应体和状态码返回，便于 debug
        return False, f"check existence failed: {r.status_code} {r.text}"

    try:
        resp = session.put(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.SSLError as e:
        return False, f"ssl error on put: {e}"
    except requests.exceptions.ReadTimeout:
        return False, "timeout on put request"
    except requests.exceptions.ConnectionError as e:
        return False, f"connection error on put: {e}"
    except Exception as e:
        return False, f"unexpected error on put: {e}"

    # 返回详细信息，方便定位
    if resp.status_code in (200, 201):
        try:
            return True, resp.json()
        except Exception:
            return True, {"status_code": resp.status_code, "text": resp.text}
    else:
        return False, f"upload failed: {resp.status_code} {resp.text}"


def add_bg_from_local(image_path="static/bg.png", sidebar_cover_rgba="rgba(0,0,0,1.0)", main_overlay_rgba="rgba(255,255,255,0.0)"):
    """
    将本地图片内联为 base64 并通过 CSS 设置为 app 主区域背景。
    参数:
      image_path: 相对路径到图片（你指定的 static/bg.png）
      sidebar_cover_rgba: 侧边栏的背景覆盖色（默认接近不透明白）
      main_overlay_rgba: 主内容区的可选遮罩（可用于调暗/调亮，默认透明）
    """
    img_data_uri = None
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode()
            # 尝试根据后缀决定 mime
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png"
            if ext in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            elif ext == ".svg":
                mime = "image/svg+xml"
            img_data_uri = f"data:{mime};base64,{encoded}"
        except Exception:
            img_data_uri = None

    # 如果没有内联成功，回退到静态路径（某些部署会把 static 目录暴露）
    img_url = img_data_uri or "/static/bg.png"

    # 注入 CSS：把背景应用到主 app 容器（sidebar 会被覆盖）
    css = f"""
    <style>
    /* 将背景设置到 app 的大容器上（只影响主区域） */
    [data-testid="stAppViewContainer"] {{
        background-image: url("{img_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* 给侧边栏一个覆盖的背景（避免背景图透出来） */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_cover_rgba};
        /* 微调侧边栏内边距/阴影（可选） */
        box-shadow: 2px 0 8px rgba(0,0,0,0.06);
    }}

    /* 主容器内部可以加一个可调节的半透明遮罩以提升可读性 */
    /* 这里选择对 block-container 应用一个背景遮罩（streamlit 版本差异，若无效可调整选择器） */
    .stApp .block-container {{
        background: {main_overlay_rgba};
    }}

    /* 保证部分组件仍然是白底（以便可读性），如果想要透明可去掉 */
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
# ---------- 工具函数：解析上传文件 ----------
def extract_text_from_file(filepath):
    """支持 txt / pdf / docx 格式，失败则返回空字符串"""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
            except Exception:
                st.error("解析 PDF 需要 PyPDF2， 请在 requirements.txt 中添加 PyPDF2 并重启。")
                return ""
            text = []
            reader = PdfReader(filepath)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(text)
        elif ext == ".docx":
            try:
                import docx
            except Exception:
                st.error("解析 docx 需要 python-docx， 请在 requirements.txt 中添加 python-docx 并重启。")
                return ""
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return ""
    except Exception as e:
        st.error(f"解析文件失败: {e}")
        return ""

# ---------- 重建索引（从 DOCS_DIR 中的所有文件） ----------
# def rebuild_vector_index(embedding):
#     """从 DOCS_DIR 的所有文件读取文本，创建/覆盖 Chroma 向量库"""
#     files = [os.path.join(DOCS_DIR, fn) for fn in os.listdir(DOCS_DIR)]
#     texts = []
#     metadatas = []
#     for fp in files:
#         txt = extract_text_from_file(fp)
#         if txt and txt.strip():
#             texts.append(txt)
#             metadatas.append({"source": os.path.basename(fp)})
#     if not texts:
#         # 如果没有文本，创建一个空的 Chroma（或返回 None）
#         # 这里我们返回 None 表示没有有效索引
#         return None

#     # 使用 Chroma.from_texts 来重建索引并持久化
#     try:
#         vectordb = Chroma.from_texts(
#             texts=texts,
#             embedding=embedding,
#             persist_directory=PERSIST_DIR,
#             metadatas=metadatas,
#         )
#         vectordb.persist()
#         return vectordb
#     except Exception as e:
#         st.error(f"重建向量索引失败: {e}")
#         return None
def rebuild_vector_index(embedding):
    """从 DOCS_DIR 的所有文件读取文本，创建/覆盖 Chroma 向量库。返回 vectordb 或 None。"""
    if not os.path.isdir(DOCS_DIR):
        st.warning(f"DOCS_DIR 不存在: {DOCS_DIR}")
        return None

    files = [fn for fn in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, fn))]
    if not files:
        st.info("DOCS_DIR 为空。")
        return None

    texts = []
    metadatas = []
    skipped = []
    for fn in files:
        fp = os.path.join(DOCS_DIR, fn)
        try:
            txt = extract_text_from_file(fp)
            if txt and txt.strip():
                texts.append(txt)
                metadatas.append({"source": fn})
            else:
                skipped.append((fn, "no text extracted / unsupported format"))
        except Exception as e:
            skipped.append((fn, f"extract error: {e}"))

    # 把调试信息展示到 UI，便于定位问题
    st.info(f"找到文件: {files}")
    if skipped:
        st.warning(f"以下文件未被加入索引（文件名, 原因）：{skipped}")
    st.info(f"将用于索引的文档数: {len(texts)}")

    if not texts:
        return None

    try:
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embedding,
            persist_directory=PERSIST_DIR,
            metadatas=metadatas,
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        st.error(f"重建向量索引失败: {e}")
        return None
# ---------- 获取检索器 ----------
def get_retriever():
    embedding = ZhipuAIEmbeddings()

    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding
        )
        return vectordb.as_retriever()
    except Exception:
        # 如果加载失败，重新创建
        vectordb = Chroma.from_texts(
            texts=["初始化文档"],
            embedding=embedding,
            persist_directory=PERSIST_DIR,
        )
        vectordb.persist()
        return vectordb.as_retriever()

# ---------- combine_docs ----------
def combine_docs(docs):
    
    try:
        # 如果是 dict-like 的情况
        if isinstance(docs, dict) and "context" in docs:
            return "\n\n".join(doc.page_content for doc in docs["context"])
        # 如果是 list
        return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
    except Exception:
        return ""

# ---------- 构建 QA chain（基本结构） ----------
def get_qa_history_chain(model_name="glm-4-plus", temperature=0.0, max_tokens=1024):
    retriever = get_retriever()
    llm = ZhipuaiLLM(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
    "你是一个问答任务的助手。"
    "请优先结合检索到的上下文片段回答问题。"
    "如果检索到的上下文为空或不足以回答问题，请结合你的通用知识回答。"
    "保持回答简洁明了。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs,
        ).assign(answer=qa_chain)
    return qa_history_chain

# ---------- gen_response（含检索不足回退） ----------
def gen_response(chain, input_text, chat_history, model_name, temperature, max_tokens):
    # 先判断检索结果是否充足
    retriever = get_retriever()
    docs = []
    try:
        docs = retriever.get_relevant_documents(input_text)
    except Exception:
        docs = []

    total_len = sum(len(getattr(d, "page_content", "") or "") for d in docs)
    # 如果检索结果不足，回退到纯 LLM（不使用检索上下文）
    if not docs or total_len < 50:
        # 调用纯 LLM 回答（同步、一次性）
        llm = ZhipuaiLLM(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        try:
            # 假定 ZhipuaiLLM 支持 invoke 或 __call__（与你的 zhipuai_llm 实现兼容）
            resp = llm.invoke([SystemMessage(content="你是一个知识助理，请结合你的通用知识简洁回答用户问题。"),
                               HumanMessage(content=input_text)])
            if hasattr(resp, "content"):
                yield resp.content
            else:
                yield str(resp)
        except Exception as e:
            yield f"调用 LLM 回退失败: {e}"
        return

    # 否则走原先的流式 RAG chain
    response = chain.stream({
        "input": input_text,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="RAG Chat with Upload", layout="wide")
    # 在页面初始化时添加背景
    add_bg_from_local(image_path="static/bg.png",
                      sidebar_cover_rgba="rgba(0,0,0,1.0)",
                      main_overlay_rgba="rgba(255,255,255,0.0)")
    st.title("🔎 基于RAG的云端个人知识库助手 🦜")

    # 左侧：参数与上传
    with st.sidebar:
        st.header("模型参数")
        model_name = st.selectbox("Model", options=['GLM-4.5-Air', 'GLM-4.7-FlashX',"glm-4.6V", "glm-4.7", "glm-4-plus", "glm-5",'GLM-4.5'], index=0)
        temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        max_tokens = st.number_input("max_tokens", min_value=64, max_value=4096, value=1024, step=64)
        top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        st.markdown("---")

        st.header("文档管理（上传/重建索引）")
        # 读取 GitHub 配置（放在 sidebar 内）
        github_token = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        github_repo = st.secrets.get("GITHUB_REPO") or os.environ.get("GITHUB_REPO")  # 格式: owner/repo
        github_branch = st.secrets.get("GITHUB_BRANCH") or os.environ.get("GITHUB_BRANCH") or "main"

        uploaded = st.file_uploader("上传本地文档（.txt, .pdf, .docx）", accept_multiple_files=True)

        if uploaded:
            for up in uploaded:
                save_path = os.path.join(DOCS_DIR, up.name)

                # 保存到本地（供当前服务器读取）
                with open(save_path, "wb") as f:
                    f.write(up.getbuffer())

                st.success(f"已保存到服务器: {up.name}")

                # ---------- 推送到 GitHub ----------
                if github_token and github_repo:
                    try:
                        with open(save_path, "rb") as f:
                            file_bytes = f.read()

                        # 可配置的单文件大小上限（防止上传太大导致连接中断）
                        MAX_BYTES = 8 * 1024 * 1024  # 8MB
                        if len(file_bytes) > MAX_BYTES:
                            st.error(f"文件过大 ({len(file_bytes)} bytes)。请使用 git push 或启用 Git LFS。")
                            continue

                        repo_path = f"data_base/docs/{up.name}"  # GitHub 中保存路径
                        commit_msg = f"Upload file {up.name} from Streamlit App"

                        ok, result = github_put_file(
                            github_token,
                            github_repo,
                            repo_path,
                            file_bytes,
                            commit_msg,
                            branch=github_branch
                        )

                        if ok:
                            st.success(f"已推送到 GitHub: {github_repo}/{repo_path}")
                        else:
                            # result 会包含失败原因或响应体，显示给用户便于 debug
                            st.error(f"GitHub 推送失败: {result}")

                    except Exception as e:
                        st.error(f"GitHub 上传异常(捕获): {e}")
                else:
                    st.warning("未配置 GITHUB_TOKEN 或 GITHUB_REPO，跳过 GitHub 推送。")

            st.info("上传完成，如需加入检索库请点击“重建索引”。")

        force_rebuild = st.button("🔁 重建索引 更新检索库")
        if force_rebuild:
            with st.spinner("正在重建索引..."):
                emb = ZhipuAIEmbeddings()
                vectordb = rebuild_vector_index(emb)
                if vectordb:
                    st.success("索引重建完成。")
                else:
                    st.warning("索引目录中没有有效文档，未生成索引。")


    # 中央：对话区
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    messages_container = st.container()

    # 显示历史消息
    for role, content in st.session_state.messages:
        with messages_container.chat_message(role):
            st.write(content)

    # 输入区
    prompt = st.chat_input("Say something")
    if prompt:
        # 先在界面显示用户消息（并暂不写入 session_state）
        with messages_container.chat_message("human"):
            st.write(prompt)

        # 构造 chat_history（不包含当前 prompt，避免重复）
        chat_history = st.session_state.messages.copy()

        # 重新创建 qa chain（确保使用最新模型参数）
        st.session_state.qa_history_chain = get_qa_history_chain(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        answer_generator = gen_response(
            chain=st.session_state.qa_history_chain,
            input_text=prompt,
            chat_history=chat_history,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 流式显示 AI 回复
        with messages_container.chat_message("ai"):
            placeholder = st.empty()
            output_chunks = []
            try:
                for chunk in answer_generator:
                    output_chunks.append(chunk)
                    # 使用 markdown 显示（也可改为 placeholder.text / info 等）
                    placeholder.markdown("".join(output_chunks))
                output = "".join(output_chunks)
                status_msg = "✅ 调用成功"
            except Exception as e:
                output = ""
                status_msg = f"❌ 调用失败: {e}"
                placeholder.markdown(status_msg)

            st.caption(status_msg)

        # 最后把问答加入 session state
        st.session_state.messages.append(("human", prompt))
        st.session_state.messages.append(("ai", output or status_msg))

if __name__ == "__main__":

    main()

