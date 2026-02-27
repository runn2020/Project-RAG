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

# 文件/路径定义
PERSIST_DIR = "data_base/vector_db2/chroma"
DOCS_DIR = "data_base/docs"

os.makedirs(DOCS_DIR, exist_ok=True)

# 从 secrets 中读取 API key（请在 Streamlit Cloud 的 Secrets 中设置）
if "ZHIPUAI_API_KEY" in st.secrets:
    os.environ['ZHIPUAI_API_KEY'] = st.secrets["ZHIPUAI_API_KEY"]
else:
    # 本地测试时可以临时设置环境变量，但不要把密钥写死到代码里
    if "ZHIPUAI_API_KEY" in os.environ:
        pass
    else:
        st.warning("未找到 ZHIPUAI_API_KEY， 请在 Streamlit Secrets 中添加。")

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
def rebuild_vector_index(embedding):
    """从 DOCS_DIR 的所有文件读取文本，创建/覆盖 Chroma 向量库"""
    files = [os.path.join(DOCS_DIR, fn) for fn in os.listdir(DOCS_DIR)]
    texts = []
    metadatas = []
    for fp in files:
        txt = extract_text_from_file(fp)
        if txt and txt.strip():
            texts.append(txt)
            metadatas.append({"source": os.path.basename(fp)})
    if not texts:
        # 如果没有文本，创建一个空的 Chroma（或返回 None）
        # 这里我们返回 None 表示没有有效索引
        return None

    # 使用 Chroma.from_texts 来重建索引并持久化
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
def get_retriever(force_reload=False):
    embedding = ZhipuAIEmbeddings()
    # 如果请求强制重建或索引不存在，则尝试从 DOCS_DIR 重建
    if force_reload or not os.path.exists(os.path.join(PERSIST_DIR, "index.sqlite")):
        vectordb = rebuild_vector_index(embedding)
        if vectordb is None:
            # 没有可用的向量库，返回一个“空”的可调用 retriever（兼容接口）
            class DummyRetriever:
                def get_relevant_documents(self, q):
                    return []
                def as_retriever(self):
                    return self
            return DummyRetriever()
        return vectordb.as_retriever()
    else:
        # 加载已有数据库
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
        return vectordb.as_retriever()

# ---------- combine_docs（与你原始的保持一致） ----------
def combine_docs(docs):
    # docs 可能是一个 list of Documents 或者一个 dict 包含 "context"
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
    "如果检索到的上下文为空或不足以回答问题，请基于你的通用知识回答。"
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
    # 先快速判断检索结果是否充足（避免始终走 RAG 导致“不知道”）
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
            resp = llm.invoke([SystemMessage(content="你是一个助理，请基于你的通用知识简洁回答用户问题。"),
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
    st.title("🔎 RAG 问答 — 带参数 & 本地文档上传")

    # 左侧：参数与上传
    with st.sidebar:
        st.header("模型参数")
        model_name = st.selectbox("Model", options=["glm-4.7", "glm-3.5"], index=0)
        temperature = st.slider("temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        max_tokens = st.number_input("max_tokens", min_value=64, max_value=4096, value=1024, step=64)
        top_p = st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
        st.markdown("---")

        st.header("文档管理（上传/重建索引）")
        uploaded = st.file_uploader("上传本地文档（.txt, .pdf, .docx）", accept_multiple_files=True)
        if uploaded:
            for up in uploaded:
                save_path = os.path.join(DOCS_DIR, up.name)
                # 保存文件
                with open(save_path, "wb") as f:
                    f.write(up.getbuffer())
                st.success(f"已保存: {up.name}")
            st.info("上传完毕，请点击下方“重建索引”以把新文档加入检索库。")

        force_rebuild = st.button("🔁 重建索引（从 data_base/docs/ 重建，会覆盖原索引）")
        if force_rebuild:
            with st.spinner("正在重建索引..."):
                emb = ZhipuAIEmbeddings()
                vectordb = rebuild_vector_index(emb)
                if vectordb:
                    st.success("索引重建完成。")
                else:
                    st.warning("索引目录中没有有效文档，未生成索引。")

        st.markdown("---")
        st.caption("请将 API key 放在 Streamlit Secrets：ZHIPUAI_API_KEY")

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