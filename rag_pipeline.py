from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

def create_kb():
    # split knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source":"langchain_knowledge_base.md"})
    chunks = splitter.split_documents([doc])
    
    # create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model, 
        persist_directory=tempfile.mkdtemp()
    )
    return vector_store



def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
    llm = init_chat_model(
        model="google_genai:gemini-2.5-flash",
        temperature=0.2
    )

    # RAG prompt template
    prompt = ChatPromptTemplate.from_template("""Use only the following pieces of context to answer the question. 
                                               If the answer is not in the context, say I don't know.
                                               Context: {context}
                                               Question: {question}""")

    # Response formatter
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # Test the RAG chain
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?"
    ]

    print("Basic RAG Demo:\n")

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"\nQuestion: {q}")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()
    
    
    

    
    
    
    
