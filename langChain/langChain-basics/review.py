from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


######## Using Ollama Chat Model

# 1. LLM 객체 생성
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# 2. 질문 정의
food_prompt = PromptTemplate(
  template='what is one of the most popular fod in {country}? Please return the name of the food only.',
  input_variables=["country"]
)

# 3. LangChain Runnable  Chain 만들기
food_chain = food_prompt | llm | StrOutputParser()

# 4. final_chain 결과 출력
food_chain.invoke({"country": "Italy"}) # 예: "Pizza"

# 5. recipe_prompt 정의
recipe_prompt = ChatPromptTemplate.from_messages([
  ("system", "Provide the rescipe of the food that the user wants Please return only the recipe without any additional text."),
  ("human", "Can you give me the recipe for making {food}?"),
])

# 6. LangChain Runnable  Chain 만들기
recipe_chain = recipe_prompt | llm | StrOutputParser()

# 7. final_chain 만들기
final_chain = {"food": food_chain} | recipe_chain

# 8. final_chain 결과 출력
print(final_chain.invoke({'country': "South Korea"})) # 예: "To make Pizza, you will need..."