from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def get_ai_message(user_message):
    
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    index_name = 'tax-markdown-index'

    # Pinecone Vector DB 연결
    database = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name,
    )
    
    llm = ChatOpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_template("""
    [Identity]
    - 당신은 최고의 한국 세무 전문가입니다.
    - [Context]를 참고하여 사용자의 질문에 답변하세요.

    [Context]
    {context}

    [User Question]
    {input}
    """)

    # LLM과 Prompt 준비
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # database는 PineconeVectorStore 같은 Retriever로 변환 가능해야 함
    retriever = database.as_retriever(search_kwargs={"k":3})

    # Retrieval + Combine 연결
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 질문을 명확하게 변경하는 체인
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    # prompt 생성
    prompt = ChatPromptTemplate.from_template(f"""

    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 사용자의 질문이 이미 명확하다면, 그대로 반환해주세요.
                                            
    [사전] : {dictionary}

    [사용자 질문] : {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    tax_chain = {"input" : dictionary_chain} | rag_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message["answer"]