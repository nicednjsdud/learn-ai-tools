# Vector Database

## 사용자가 원하는 정보

### 1. 사용자의 질문과 관련있는 데이터

-   관련이 있다는 것을 어떻게 판단할까?
-   관련성 파악을 위해 vector를 활용함
    -   단어 또는 문장의 유사도를 파악해서 관련성을 측정

### 2. Vector를 생성하는 방법

-   Embedding 모델을 활용해서 vector를 생성함
-   문장에서 비슷한 단어가 자주 붙어있는 것을 학습
    -   왕은 왕자의 아버지다.
    -   여왕은 왕자의 어머니다.
    -   "왕자의"라는 단어 앞에 등장하는 "왕"과 "여왕"은 유사할 가능성이
        높다.
    -   Embedding Projector

## Vector Database

### 1. Embedding 모델을 활용해 생성된 vector를 저장

-   단순히 vector만 저장하면 안되고 metadata도 같이 저장
    -   문서의 이름, 페이지 번호 등등을 같이 저장 → LLM이 생성하는
        답변의 퀄리티가 상승함

### 2. Vector를 대상으로 유사도 검색 실시

-   사용자의 질문과 가장 비슷한 문서를 가져오는 것 - **Retrieval**
-   가져온 문서를 prompt를 통해 LLM에 제공 - **Augmented**
-   LLM은 prompt를 활용해서 답변 생성 - **Generation**

------------------------------------------------------------------------

## 📚 출처

-   [Inflearn - RAG LLM 어플리케이션 개발
    (LangChain)](https://www.inflearn.com/course/rag-llm-application%EA%B0%9C%EB%B0%9C-langchain/community)
