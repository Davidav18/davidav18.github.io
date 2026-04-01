import bs4
import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import Literal

# --- 1. LLM & LANGCHAIN IMPORTS ---
from pydantic import BaseModel, Field 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 2. CONFIGURACIÓN DE ENTORNO
# ==========================================
load_dotenv()

llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
    openai_api_base='https://api.deepseek.com',
    temperature=0
)

# ==========================================
# 3. ETAPA DE INGESTIÓN (ETL)
# ==========================================
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

prompt_rag_template = "Answer the question based only on the following context:\n{context}\nQuestion: {question}"
prompt_rag = ChatPromptTemplate.from_template(prompt_rag_template)

# ==========================================
# 4.1 LÓGICA DE DESCOMPOSICIÓN
# ==========================================
template_decomposition = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# ==========================================
# 4.2 LÓGICA DE STEP-BACK PROMPTING
# ==========================================
examples = [
    {"input": "Could the members of The Police perform lawful arrests?", "output": "what can the members of The Police do?"},
    {"input": "Jan Sindel's was born in what country?", "output": "what is Jan Sindel's personal history?"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question."),
    few_shot_prompt,
    ("user", "{question}"),
])

generate_step_back_query = step_back_prompt | llm | StrOutputParser()

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. 
Your response should be comprehensive and use the provided context from a general step-back question 
as well as the specific user query.

Step-back Context:
{step_back_context}

Specific Context:
{context}

Original Question: {question}
Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain_step_back_final = (
    {
        "step_back_context": generate_step_back_query | retriever,
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4.3 LÓGICA DE HyDE
# ==========================================
template_hyde = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template_hyde)

generate_docs_for_retrieval = (
    prompt_hyde 
    | llm 
    | StrOutputParser() 
)

hyde_retrieval_chain = generate_docs_for_retrieval | retriever 

# ==========================================
# 4.4 LÓGICA DE ROUTING (VERSIÓN COMPATIBLE DEEPSEEK)
# ==========================================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

parser_router = JsonOutputParser(pydantic_object=RouteQuery)

system_router = """You are an expert at routing a user question to a vectorstore or web_search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use web_search for anything else.

{format_instructions}"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_router),
    ("human", "{question}"),
])

question_router = (
    route_prompt.partial(format_instructions=parser_router.get_format_instructions()) 
    | llm 
    | parser_router
)

# ==========================================
# 5. FUNCIONES DE APOYO
# ==========================================
def format_qa_pairs(questions, answers):
    formatted = ""
    for i, (q, a) in enumerate(zip(questions, answers), start=1):
        formatted += f"Question {i}: {q}\nAnswer {i}: {a}\n\n"
    return formatted.strip()

# ==========================================
# 6. EJECUCIÓN (FLUJO CON ROUTING CORREGIDO)
# ==========================================
if __name__ == "__main__":
    # Prueba 1: Pregunta del blog
    question = "What are the main components of an LLM-powered autonomous agent system?"
    
    # Descomenta la siguiente línea para probar el camino de web_search:
    # question = "Who won the last football world cup?"

    print(f"\n--- 🚦 ANALIZANDO PREGUNTA: {question} ---")
    
    # 1. El Router decide el camino
    source = question_router.invoke({"question": question})
    
    # CORRECCIÓN: Acceso por llave de diccionario
    print(f"📍 Ruta seleccionada por el LLM: {source['datasource']}")

    if source['datasource'] == "vectorstore":
        print("🔍 Ejecutando pipeline RAG avanzado (HyDE)...")
        
        retrieved_docs_hyde = hyde_retrieval_chain.invoke({"question": question})
        final_answer = (prompt_rag | llm | StrOutputParser()).invoke({
            "context": retrieved_docs_hyde, 
            "question": question
        })
        print(f"\n✅ RESPUESTA FINAL (RAG):\n{final_answer}")
        
    else:
        print("🌐 La pregunta no es sobre el blog. Redirigiendo a respuesta general...")
        general_answer = llm.invoke(question)
        print(f"\n✅ RESPUESTA GENERAL:\n{general_answer.content}")