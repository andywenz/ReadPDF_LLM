import os
import sys
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pickle
from tqdm import tqdm
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

class PDFProcessor:
    def __init__(self, base_url=OLLAMA_BASE_URL, model_name=MODEL_NAME):
        self.base_url = base_url
        self.model_name = model_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.model_name
        )
        self.llm = OllamaLLM(
            base_url=self.base_url,
            model=self.model_name,
            temperature=0.5
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件提取文本内容"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            text_content = []
            
            # 使用tqdm创建进度条
            print(f"正在提取PDF文本 ({total_pages} 页)...")
            for page_num in tqdm(range(total_pages), desc="PDF提取进度"):
                page = doc.load_page(page_num)
                text_content.append(page.get_text())
                
            doc.close()
            return "\n".join(text_content)
            
        except Exception as e:
            print(f"提取PDF内容时出错: {str(e)}")
            return ""

    def process_pdf(self, pdf_path: str, save_index: bool = True) -> FAISS:
        """处理PDF并创建向量索引"""
        # 获取PDF文件名(不含扩展名)，用于索引文件命名
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        index_dir = f"{pdf_name}_faiss_index"
        
        # 检查是否已存在索引目录
        if os.path.exists(index_dir) and save_index:
            print(f"找到现有索引目录 {index_dir}，正在加载...")
            try:
                vectorstore = FAISS.load_local(index_dir, self.embeddings)
                print("索引加载成功！")
                return vectorstore
            except Exception as e:
                print(f"加载索引文件失败: {str(e)}")
                print("将重新生成索引...")
        
        # 提取PDF文本
        text_content = self.extract_text_from_pdf(pdf_path)
        if not text_content:
            raise ValueError("无法从PDF中提取文本内容")
        
        # 分割文本
        print("正在分割文本...")
        splits = self.text_splitter.split_text(text_content)
        print(f"文本已分割为 {len(splits)} 个片段")
        
        # 创建向量存储
        print("正在生成文本嵌入并创建FAISS索引...")
        with tqdm(total=len(splits), desc="嵌入生成进度") as pbar:
            vectorstore = self._create_vectorstore_with_progress(splits, pbar)
        
        # 保存索引文件
        if save_index:
            print(f"正在保存索引到 {index_dir} 目录...")
            try:
                # 使用FAISS的本地保存方法而不是pickle
                vectorstore.save_local(index_dir)
                print("索引保存成功！")
            except Exception as e:
                print(f"保存索引时出错: {str(e)}")
                print("将继续使用内存中的索引...")
            
        return vectorstore
    
    def _create_vectorstore_with_progress(self, texts: List[str], pbar: tqdm) -> FAISS:
        """创建向量存储并显示进度"""
        # 因为FAISS.from_texts不直接支持进度条，我们通过批处理实现
        batch_size = 10  # 可以根据需要调整批大小
        
        # 创建FAISS向量库
        if len(texts) > 0:
            # 处理第一批次来初始化向量库
            first_batch = texts[:batch_size]
            vectorstore = FAISS.from_texts(first_batch, self.embeddings)
            pbar.update(len(first_batch))
            
            # 处理剩余批次
            for i in range(batch_size, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch = texts[i:end_idx]
                
                # 将当前批次添加到向量库
                vectorstore.add_texts(batch)
                pbar.update(len(batch))
                
            return vectorstore
        else:
            raise ValueError("没有文本可以向量化")
        
    def create_rag_system(self, vectorstore: FAISS):
        """创建RAG检索增强生成系统"""
        # 创建检索器
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回3个最相关的文档片段
        )
        
        # 创建提示模板
        template = """你是一个基于文档回答问题的助手。使用以下检索到的文档片段来回答问题。如果你无法从文档中找到答案，请说 "我无法从提供的文档中找到答案"。
        提供的文档:{context}
        问题: {question}
        请给出详细的回答，如果可能，引用文档中的具体内容:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建 RAG 链
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

def main():
    print("=" * 50)
    print("PDF文档问答系统 (基于Ollama和FAISS)")
    print("=" * 50)
    
    processor = PDFProcessor()
    
    # 获取PDF路径
    while True:
        pdf_path = input("\n请输入PDF文件的完整路径: ").strip()
        
        if pdf_path.lower() == 'exit':
            print("感谢使用，再见！")
            sys.exit(0)
            
        if not os.path.exists(pdf_path):
            print(f"错误: 文件 '{pdf_path}' 不存在，请重新输入。")
            continue
            
        if not pdf_path.lower().endswith('.pdf'):
            print(f"错误: 文件 '{pdf_path}' 不是PDF文件，请重新输入。")
            continue
            
        break
    
    try:
        # 处理PDF并创建向量存储
        vectorstore = processor.process_pdf(pdf_path)
        
        # 创建RAG系统
        rag_chain = processor.create_rag_system(vectorstore)
        
        print("\n" + "*" * 50)
        print(f"PDF文档 '{os.path.basename(pdf_path)}' 已成功加载！")
        print("您可以开始提问关于该PDF内容的问题。")
        print("输入 'exit' 退出程序，输入 'new' 加载新的PDF文件。")
        print("*" * 50 + "\n")
        
        # 问答循环
        while True:
            question = input("\n请输入您的问题: ")
            
            if question.lower() == 'exit':
                print("感谢使用，再见！")
                break
                
            if question.lower() == 'new':
                print("\n准备加载新的PDF文件...")
                main()  # 重新启动主程序
                break
            
            try:
                # 显示思考中的提示
                print("思考中...", end="", flush=True)
                start_time = time.time()
                
                # 调用RAG链获取回答
                response = rag_chain.invoke(question)
                
                # 清除思考中的提示
                print("\r" + " " * 20 + "\r", end="", flush=True)
                
                # 显示回答
                processing_time = time.time() - start_time
                print(f"\n回答: (处理时间: {processing_time:.2f}秒)")
                print("-" * 50)
                print(textwrap.fill(response, width=80))
                print("-" * 50)
                
            except Exception as e:
                print(f"\n处理问题时出错: {str(e)}")
                
    except Exception as e:
        print(f"运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息帮助调试
        
if __name__ == "__main__":
    main()