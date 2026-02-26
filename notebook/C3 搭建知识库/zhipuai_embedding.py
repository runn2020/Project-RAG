from typing import List
from langchain_core.embeddings import Embeddings
import os

class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    
    def __init__(self, api_key: str = None):
        """
        实例化ZhipuAI
        
        Args:
            api_key: 智谱AI的API密钥，如果不提供则从环境变量ZHIPUAI_API_KEY读取
        """
        from zhipuai import ZhipuAI
        
        # 获取 API Key：优先使用传入的，否则从环境变量读取
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        
        # 尝试从 Streamlit secrets 读取（如果是Streamlit环境）
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get('ZHIPUAI_API_KEY')
            except:
                pass
        
        if not self.api_key:
            raise ValueError("请提供api_key，可以通过参数、环境变量ZHIPUAI_API_KEY或Streamlit secrets提供")
        
        self.client = ZhipuAI(api_key=self.api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        result = []
        for i in range(0, len(texts), 64):
            embeddings = self.client.embeddings.create(
                model="embedding-3",
                input=texts[i:i+64]
            )
            result.extend([embeddings.embedding for embeddings in embeddings.data])
        return result
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        return self.embed_documents([text])[0]
