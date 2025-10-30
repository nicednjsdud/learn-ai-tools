from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
######## Using Ollama Chat Model

chat_ollama = ChatOllama(model="llama3.2:1b")

### Direct Prompt
response = chat_ollama.invoke("What is the capital of France?")  ## Prompt -> LLM 호출 명령어

# print(response)


######## Using PromptTemplate

prompt_template = PromptTemplate(
  template="What is the capital of {country}?",
  input_variables=["country"]
)

## Prompt 생성
prompt = prompt_template.invoke({"country": "France"})

## LLM 호출
response_with_prompt = chat_ollama.invoke(prompt)

# print(response_with_prompt)


######## Using Messages

message_list = [
  SystemMessage(content="You are a helpful assistant."),
  AIMessage(content="The capital of France is Paris."),
  HumanMessage(content="What is the capital of France?"),
]


## LLM 호출
response_with_message = chat_ollama.invoke(message_list)

print(response_with_message)