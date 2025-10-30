# from langchain_ollama import ChatOllama
# # from langchain_openai import ChatOpenAI
# # from langchain_openai import AzureOpenAI
# from langchain_anthropic import ChatAnthropic

# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from a .env file

# ######## Using Ollama Chat Model

# chat_ollama = ChatOllama(model="llama3.2:1b")

# chat_ollama.invoke("What is the capital of France?")

# response = chat_ollama.invoke("What is the capital of France?")

# print(response)

# ######### Using OpenAI Chat Model

# chat_openai = ChatOpenAI(model="gpt-4o-mini")

# response_openai = chat_openai.invoke("What is the capital of France?") # You need to set the OPENAI_API_KEY environment variable for this to work.

# print(response_openai)



# ########## Using Azure OpenAI Chat Model

# chat_azure = AzureOpenAI(model="gpt-4o-mini")

# response_azure = chat_azure.invoke("What is the capital of France?")

# print(response_azure)  # You need to set the AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables for this to work.


# ########## Using Anthropic Claude Chat Model

# chat_anthropic = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# response_anthropic = chat_anthropic.invoke("What is the capital of France?")  # You need to set the ANTHROPIC_API_KEY environment variable for this to work.

# print(response_anthropic)

