from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
######## Using Ollama Chat Model

llm = ChatOllama(model="llama3.2:1b")

### Direct Prompt
response = llm.invoke("What is the capital of France?")  ## Prompt -> LLM 호출 명령어

# print(response)


######## Using PromptTemplate

prompt_template = PromptTemplate(
  template="What is the capital of {country}? Return the name of the capital only.",
  input_variables=["country"]
)

## Prompt 생성
prompt = prompt_template.invoke({"country": "France"})

## LLM 호출
response_with_prompt = llm.invoke(prompt)

# output 
output_parser = StrOutputParser()

## 파싱
responseParser = output_parser.invoke(response_with_prompt)

print(responseParser) # Paris


### Using Pydantic Output Parser JSON 형태로 응답받아 파싱하기

class CountryDetails(BaseModel):
    capital: str = Field(..., description="The capital city of the country")
    population: int = Field(..., description="The population of the capital city")
    language: str = Field(..., description="The primary language spoken in the country")
    currency: str = Field(..., description="The currency used in the country")

structured_llm = llm.with_structured_output(CountryDetails)

response_structured = structured_llm.invoke(prompt_template.invoke({"country": "France"}))

print(response_structured) # capital='Paris' population=21400000 language='français' currency='euro'