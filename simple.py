import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import textwrap

# 配置参数
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2"
DOCUMENT_TEXT = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于开发能够模拟人类智能的系统和软件。这些系统能够执行通常需要人类智能的任务，如视觉感知、语音识别、决策和语言翻译。

人工智能的发展可以追溯到20世纪50年代，但在近年来由于计算能力的提升、大数据的可用性以及算法的进步而取得了显著的进展。如今，人工智能已经融入了我们日常生活的方方面面，从智能手机中的语音助手到电子商务网站上的推荐系统。

机器学习是人工智能的一个重要子集，它使计算机系统能够通过经验自动改进。深度学习则是机器学习的一种特殊形式，它使用神经网络在大量数据上进行训练，能够在图像识别、自然语言处理等领域取得令人印象深刻的成果。

尽管人工智能技术带来了许多便利和创新，但它也引发了关于隐私、安全、就业以及伦理决策等方面的担忧和讨论。如何负责任地开发和使用人工智能成为了当今社会面临的重要课题。
"""

def create_rag_system():
    # 初始化 Ollama 模型实例
    llm = OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=0.5
    )
    
    # 初始化 Ollama 嵌入模型
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME
    )

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # 分割文档
    splits = text_splitter.split_text(DOCUMENT_TEXT)
    
    # 使用 Ollama 嵌入和 FAISS 创建向量存储
    vectorstore = FAISS.from_texts(
        texts=splits,
        embedding=embeddings
    )
    
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # 返回2个最相关的文档片段
    )
    
    # 创建提示模板
    template = """你是一个基于文档回答问题的助手。使用以下检索到的文档片段来回答问题。如果你无法从文档中找到答案，请说 "我无法从提供的文档中找到答案"。
    提供的文档:{context}
    问题: {question}
    请给出详细的回答:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 构建 RAG 链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def main():
    print("正在初始化 RAG 系统...")
    rag_chain = create_rag_system()
    
    print("\n" + "*"*50)
    print("文档已加载完成，您可以开始提问有关人工智能的问题。")
    print("输入 'exit' 退出程序。")
    print("*"*50 + "\n")
    
    while True:
        question = input("\n请输入您的问题: ")
        if question.lower() == 'exit':
            print("感谢使用，再见！")
            break
        
        try:
            response = rag_chain.invoke(question)
            print("\n回答:")
            print(textwrap.fill(response, width=80))
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()