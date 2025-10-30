from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


######## Using Ollama Chat Model

# 1. LLM 객체 생성
llm = ChatOllama(model="llama3.2:1b")

# 2. output 
output_parser = StrOutputParser()

# 3. capital_template 사용
capital_chain = PromptTemplate(
  template="What is the capital of {country}? Return the name of the capital only.",
  input_variables=["country"]
)

# 4. create Runnable Chain
capital_chain = capital_chain | llm | StrOutputParser()

# 5. capital_chain 호출
capital_chain.invoke({"country": "France"})


# 6. country_template 사용
country_template = PromptTemplate(
  template="Guess the name of the country based on the following information:" \
  "{information}" \
  " Return the name of the country only."
  "",
  input_variables=["information"]
)

# 7. LangChain Runnable  Chain 만들기
country_chain = country_template | llm | output_parser

# 8. country_chain 호출
country_chain.invoke({"information": "The country is very famous for its wine in Europe"})  

# 9. final_chain 만들기
final_chain ={"information" : RunnablePassthrough()} | {"country": country_chain} | capital_chain

# 10. final_chain 결과 출력
print(final_chain.invoke("The country is very famous for its wine in Europe"))

