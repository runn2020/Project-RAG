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

# ---------- GitHub <-> Chroma 合并工具（下载 -> 备份 -> 合并 -> 上传 -> 回滚点） ----------
import tempfile
import shutil
import zipfile
import hashlib
import time
import json

# 最大单文件上传阈值（需与 github_put_file 内 MAX_BYTES 保持一致）
GITHUB_UPLOAD_MAX_BYTES = 8 * 1024 * 1024  # 8 MB

def _make_tmpdir(prefix="chroma_merge_"):
    return tempfile.mkdtemp(prefix=prefix)

def _zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.relpath(full, src_dir)
                zf.write(full, arcname)

def download_github_dir(token, repo_full, dir_in_repo, save_to_dir, branch="main", timeout=30):
    """
    递归下载 repo:dir_in_repo 到本地 save_to_dir（保留结构）。
    返回 (True, "ok") 或 (False, "err_msg")
    """
    session = requests.Session()
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    api_url = f"https://api.github.com/repos/{repo_full}/contents/{dir_in_repo.lstrip('/')}"
    try:
        r = session.get(api_url, headers=headers, params={"ref": branch}, timeout=timeout)
    except Exception as e:
        return False, f"请求 GitHub 列表失败: {e}"

    if r.status_code != 200:
        return False, f"list failed: {r.status_code} {r.text}"

    items = r.json()
    os.makedirs(save_to_dir, exist_ok=True)

    for item in items:
        it_type = item.get("type")
        name = item.get("name")
        if it_type == "file":
            # 尝试使用 download_url（速度快），否则使用 contents API 获取 base64
            download_url = item.get("download_url")
            target = os.path.join(save_to_dir, name)
            try:
                if download_url:
                    rr = session.get(download_url, timeout=timeout)
                    if rr.status_code == 200:
                        with open(target, "wb") as f:
                            f.write(rr.content)
                    else:
                        return False, f"下载文件失败 {download_url}: {rr.status_code}"
                else:
                    # fallback to contents API for this file
                    file_meta = session.get(item["url"], headers=headers, params={"ref": branch}, timeout=timeout)
                    if file_meta.status_code != 200:
                        return False, f"get file meta failed: {file_meta.status_code}"
                    content_b64 = file_meta.json().get("content", "")
                    data = base64.b64decode(content_b64)
                    with open(target, "wb") as f:
                        f.write(data)
            except Exception as e:
                return False, f"下载单文件失败: {e}"
        elif it_type == "dir":
            subdir = os.path.join(save_to_dir, name)
            ok, msg = download_github_dir(token, repo_full, f"{dir_in_repo.rstrip('/')}/{name}", subdir, branch, timeout)
            if not ok:
                return False, msg
        else:
            # ignore symlink/submodule etc.
            continue
    return True, "ok"

def upload_local_dir_to_github(token, repo_full, local_dir, remote_dir, branch="main", commit_prefix="merge"):
    """
    将 local_dir 下所有文件逐个上传到 repo:remote_dir（覆盖）。返回列表 (remote_path, success_bool, result)
    注意：对于大文件（> GITHUB_UPLOAD_MAX_BYTES）会跳过并返回错误信息；考虑改用 LFS 或 release asset。
    """
    results = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            local_path = os.path.join(root, fn)
            rel_path = os.path.relpath(local_path, local_dir)
            remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/").lstrip("/")
            try:
                with open(local_path, "rb") as f:
                    content = f.read()
                if len(content) > GITHUB_UPLOAD_MAX_BYTES:
                    results.append((remote_path, False, f"file too large ({len(content)} bytes). Use LFS or upload manually."))
                    continue
                msg = f"{commit_prefix}: update {remote_path}"
                ok, res = github_put_file(token, repo_full, remote_path, content, commit_message=msg, branch=branch)
                results.append((remote_path, ok, res))
            except Exception as e:
                results.append((remote_path, False, f"read/upload exception: {e}"))
    return results

# 生成要合并的 chunk（将 DOCS_DIR 中的所有文件分块并返回 chunks/metadatas）
def generate_chunks_from_docs(docs_dir, chunk_size=800, chunk_overlap=150):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    metadatas = []
    for fn in os.listdir(docs_dir):
        fp = os.path.join(docs_dir, fn)
        if not os.path.isfile(fp):
            continue
        txt = extract_text_from_file(fp)
        if not txt or not txt.strip():
            continue
        subchunks = splitter.split_text(txt)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        for i, c in enumerate(subchunks):
            chunks.append(c)
            metadatas.append({"source": fn, "chunk_index": i, "merged_at": ts})
    return chunks, metadatas

def merge_new_chunks_into_github_chroma(github_token, github_repo, chroma_repo_dir, emb_obj, docs_dir, github_branch="main"):
    """
    完整流程：
     1) 下载远端 chroma 目录到临时目录
     2) 备份（zip）并上传到 repo backups/
     3) 打开本地临时 chroma DB，追加 new_chunks，persist
     4) 将本地临时目录上传回远端（覆盖）
    返回 (True, info) 或 (False, err)
    """
    tmp_remote = _make_tmpdir("chroma_remote_")
    try:
        st.info("1) 开始下载远端 chroma 数据目录...")
        ok, msg = download_github_dir(github_token, github_repo, chroma_repo_dir, tmp_remote, branch=github_branch)
        if not ok:
            return False, f"下载远端 chroma 失败: {msg}"

        # 备份：把下载下来的目录 zip 打包并上传回仓库 backups/
        backup_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        backup_name = f"backups/chroma_backup_{backup_ts}.zip"
        local_backup_zip = os.path.join(tempfile.gettempdir(), f"chroma_backup_{backup_ts}.zip")
        _zip_dir(tmp_remote, local_backup_zip)
        # 检查大小
        if os.path.getsize(local_backup_zip) > GITHUB_UPLOAD_MAX_BYTES:
            # 备份过大，提示用户并继续（但不上传备份），建议使用 LFS 或 S3
            st.warning(f"备份 zip 太大（{os.path.getsize(local_backup_zip)} bytes），无法上传到 GitHub。请考虑使用 Git LFS 或 S3。备份已保存在服务器临时路径：{local_backup_zip}")
            backup_uploaded = False
            backup_remote_path = None
        else:
            with open(local_backup_zip, "rb") as f:
                content = f.read()
            ok_b, res_b = github_put_file(github_token, github_repo, backup_name, content, commit_message=f"backup chroma {backup_ts}", branch=github_branch)
            if not ok_b:
                return False, f"上传备份失败: {res_b}"
            backup_uploaded = True
            backup_remote_path = backup_name

        st.info("2) 备份完成，准备打开本地 chroma 并合并新的 chunk ...")

        # 打开 Chroma 指向下载的目录
        try:
            vectordb = Chroma(persist_directory=tmp_remote, embedding_function=emb_obj)
        except Exception as e:
            return False, f"打开远端 chroma 本地副本失败: {e}"

        # 生成要新增的 chunk（从 DOCS_DIR）
        new_chunks, new_mds = generate_chunks_from_docs(docs_dir)
        if not new_chunks:
            return False, "没有在 DOCS_DIR 中找到新的可索引文档或 chunk。"

        st.info(f"将要添加的 chunk 数量: {len(new_chunks)} （请注意：可能会产生重复条目，建议后续去重或在 metadata 中比对）")

        # 添加新向量（根据 Chroma 版本，调用 add_texts 或 add_documents）
        try:
            if hasattr(vectordb, "add_texts"):
                vectordb.add_texts(texts=new_chunks, metadatas=new_mds)
            else:
                # fallback: 使用 from_texts + merge 或 add_documents
                # 若没有 add_texts，尝试 add_documents（需要 Document 类型）
                try:
                    vectordb.add_documents([type("D",(object,),{"page_content":t})() for t in new_chunks])
                except Exception:
                    return False, "向量库不支持 add_texts/add_documents，无法追加。"
        except Exception as e:
            return False, f"向量追加失败: {e}"

        # 持久化
        try:
            vectordb.persist()
        except Exception as e:
            return False, f"持久化失败: {e}"

        st.info("新向量已写入本地副本，开始上传回 GitHub 覆盖远端 chroma 数据（逐文件覆盖）...")

        # 上传回 GitHub（逐文件）；会返回每个文件的上传结果
        upload_results = upload_local_dir_to_github(github_token, github_repo, tmp_remote, chroma_repo_dir, branch=github_branch, commit_prefix="merge")

        # 检查是否有失败项（例如文件过大或上传错误）
        fails = [r for r in upload_results if not r[1]]
        if fails:
            # 列出失败并返回（注意：在这种情况下远端可能已部分被覆盖）
            return False, {"uploaded_count": len([r for r in upload_results if r[1]]), "fails": fails, "backup_uploaded": backup_uploaded, "backup_remote_path": backup_remote_path}

        # 成功
        return True, {"uploaded_count": len(upload_results), "backup_uploaded": backup_uploaded, "backup_remote_path": backup_remote_path}
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_remote, ignore_errors=True)
        except Exception:
            pass

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
def rebuild_vector_index(embedding):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    if not os.path.isdir(DOCS_DIR):
        st.warning(f"DOCS_DIR 不存在: {DOCS_DIR}")
        return None

    files = [fn for fn in os.listdir(DOCS_DIR)
             if os.path.isfile(os.path.join(DOCS_DIR, fn))]

    if not files:
        st.info("DOCS_DIR 为空。")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    texts = []
    metadatas = []

    for fn in files:
        fp = os.path.join(DOCS_DIR, fn)
        txt = extract_text_from_file(fp)

        if txt and txt.strip():
            chunks = text_splitter.split_text(txt)

            for chunk in chunks:
                texts.append(chunk)
                metadatas.append({"source": fn})

    st.info(f"生成 chunk 数量: {len(texts)}")

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
# def rebuild_vector_index(embedding):
#     """从 DOCS_DIR 的所有文件读取文本，创建/覆盖 Chroma 向量库。返回 vectordb 或 None。"""
#     if not os.path.isdir(DOCS_DIR):
#         st.warning(f"DOCS_DIR 不存在: {DOCS_DIR}")
#         return None

#     files = [fn for fn in os.listdir(DOCS_DIR) if os.path.isfile(os.path.join(DOCS_DIR, fn))]
#     if not files:
#         st.info("DOCS_DIR 为空。")
#         return None

#     texts = []
#     metadatas = []
#     skipped = []
#     for fn in files:
#         fp = os.path.join(DOCS_DIR, fn)
#         try:
#             txt = extract_text_from_file(fp)
#             if txt and txt.strip():
#                 texts.append(txt)
#                 metadatas.append({"source": fn})
#             else:
#                 skipped.append((fn, "no text extracted / unsupported format"))
#         except Exception as e:
#             skipped.append((fn, f"extract error: {e}"))

#     # 把调试信息展示到 UI，便于定位问题
#     st.info(f"找到文件: {files}")
#     if skipped:
#         st.warning(f"以下文件未被加入索引（文件名, 原因）：{skipped}")
#     st.info(f"将用于索引的文档数: {len(texts)}")

#     if not texts:
#         return None

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
    "请优先结合检索到的上下文片段回答问题，并告知用户是否检索到"
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
    st.title("🔎 云端智能个人知识库助手 🦜")

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
                # 立即列出目录，便于 debug
                try:
                    files_info = [(fn, os.path.getsize(os.path.join(DOCS_DIR, fn))) for fn in os.listdir(DOCS_DIR)]
                    st.info(f"当前 {DOCS_DIR} 文件: {files_info}")
                except Exception as e:
                    st.warning(f"列出 DOCS_DIR 失败: {e}")
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

        # force_rebuild = st.button("🔁 重建索引 更新检索库")
        # if force_rebuild:
        #     with st.spinner("正在重建索引..."):
        #         emb = ZhipuAIEmbeddings()
        #         vectordb = rebuild_vector_index(emb)
        #         if vectordb:
        #             st.success("索引重建完成。")
        #         else:
        #             st.warning("索引目录中没有有效文档，未生成索引。")
        # ---------- 合并：本地重建索引 + 可选：备份并合并到 GitHub Chroma ----------
        force_rebuild = st.button("🔁 重建索引 更新检索库（并可选合并远端）")
        if force_rebuild:
            with st.spinner("正在重建索引..."):
                emb = ZhipuAIEmbeddings()
                vectordb = rebuild_vector_index(emb)
                if vectordb:
                    st.success("本地索引重建完成。")
                else:
                    st.warning("索引目录中没有有效文档，未生成索引。")
                    # 如果没有本地索引，则不继续合并
                    continue_merge = False
                # 如果要合并到 GitHub Chroma（仅在配置了 github_token & repo 时显示选项）
                if "github_token" in locals() and github_token and "github_repo" in locals() and github_repo:
                    # 给用户一个额外的确认开关（默认不合并）
                    merge_to_github = st.checkbox("同时备份并合并到 GitHub Chroma（备份 -> 合并 -> 覆盖上传）", value=False)
                else:
                    merge_to_github = False
                    if not (github_token and github_repo):
                        st.info("若要同时合并到 GitHub Chroma，请在 Streamlit Secrets 或环境变量中配置 GITHUB_TOKEN 与 GITHUB_REPO。")

                if vectordb and merge_to_github:
                    # 运行合并流程（使用之前加入的 merge_new_chunks_into_github_chroma）
                    with st.spinner("开始备份远端并合并新向量到 GitHub Chroma（可能较慢）..."):
                        try:
                            ok, info = merge_new_chunks_into_github_chroma(
                                github_token=github_token,
                                github_repo=github_repo,
                                chroma_repo_dir="data_base/vector_db2/chroma",  # 根据你仓库实际路径调整
                                emb_obj=emb,
                                docs_dir=DOCS_DIR,
                                github_branch=github_branch
                            )
                        except Exception as e:
                            ok = False
                            info = f"合并流程抛出异常: {e}"

                        if ok:
                            st.success("合并并上传到 GitHub Chroma 完成。")
                            st.write(info)
                            if isinstance(info, dict) and info.get("backup_uploaded"):
                                st.info(f"备份已上传到仓库: {info.get('backup_remote_path')}")
                            else:
                                # 若备份未上传（通常是太大），提示服务器上备份位置（merge 函数会在 UI 中提示）
                                st.info("若备份文件过大，已保存在服务器临时路径，请查看上方提示。")
                        else:
                            st.error("合并失败。")
                            st.write(info)
                            st.warning("若合并失败但上传了部分文件，仓库可能被部分覆盖。可通过备份回滚。")

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

