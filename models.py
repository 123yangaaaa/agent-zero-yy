import os
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_mistralai import ChatMistralAI
from pydantic.v1.types import SecretStr
from python.helpers.dotenv import load_dotenv
import base64
import requests
import uuid
from zhipuai import ZhipuAI

# environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.0

# Utility function to get API keys from environment variables
def get_api_key(service):
    return os.getenv(f"API_KEY_{service.upper()}") or os.getenv(f"{service.upper()}_API_KEY")

# Ollama models
def get_ollama_chat(model_name:str, temperature=DEFAULT_TEMPERATURE, base_url=os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434", num_ctx=8192):
    return ChatOllama(model=model_name,temperature=temperature, base_url=base_url, num_ctx=num_ctx)

def get_ollama_embedding(model_name:str, temperature=DEFAULT_TEMPERATURE, base_url=os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"):
    
    return OllamaEmbeddings(model=model_name,temperature=temperature, base_url=base_url)

# HuggingFace models

def get_huggingface_embedding(model_name:str):
    return HuggingFaceEmbeddings(model_name=model_name)

# LM Studio and other OpenAI compatible interfaces
def get_lmstudio_chat(model_name:str, temperature=DEFAULT_TEMPERATURE, base_url=os.getenv("LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"):
    return ChatOpenAI(model_name=model_name, base_url=base_url, temperature=temperature, api_key="none") # type: ignore

def get_lmstudio_embedding(model_name:str, base_url=os.getenv("LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1"):
    return OpenAIEmbeddings(model=model_name, api_key="none", base_url=base_url, check_embedding_ctx_length=False) # type: ignore

# Anthropic models
def get_anthropic_chat(model_name:str, api_key=get_api_key("anthropic"), temperature=DEFAULT_TEMPERATURE):
    return ChatAnthropic(model_name=model_name, temperature=temperature, api_key=api_key) # type: ignore

# OpenAI models
def get_openai_chat(model_name:str, api_key=get_api_key("openai"), temperature=DEFAULT_TEMPERATURE):
    return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key) # type: ignore

def get_openai_instruct(model_name:str, api_key=get_api_key("openai"), temperature=DEFAULT_TEMPERATURE):
    return OpenAI(model=model_name, temperature=temperature, api_key=api_key) # type: ignore

def get_openai_embedding(model_name:str, api_key=get_api_key("openai")):
    return OpenAIEmbeddings(model=model_name, api_key=api_key) # type: ignore

def get_azure_openai_chat(deployment_name:str, api_key=get_api_key("openai_azure"), temperature=DEFAULT_TEMPERATURE, azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT")):
    return AzureChatOpenAI(deployment_name=deployment_name, temperature=temperature, api_key=api_key, azure_endpoint=azure_endpoint) # type: ignore

def get_azure_openai_instruct(deployment_name:str, api_key=get_api_key("openai_azure"), temperature=DEFAULT_TEMPERATURE, azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT")):
    return AzureOpenAI(deployment_name=deployment_name, temperature=temperature, api_key=api_key, azure_endpoint=azure_endpoint) # type: ignore

def get_azure_openai_embedding(deployment_name:str, api_key=get_api_key("openai_azure"), azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT")):
    return AzureOpenAIEmbeddings(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint) # type: ignore

# Google models
def get_google_chat(model_name:str, api_key=get_api_key("google"), temperature=DEFAULT_TEMPERATURE):
    return GoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key, safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE }) # type: ignore

# Mistral models
def get_mistral_chat(model_name:str, api_key=get_api_key("mistral"), temperature=DEFAULT_TEMPERATURE):
    return ChatMistralAI(model=model_name, temperature=temperature, api_key=api_key) # type: ignore

# Groq models
def get_groq_chat(model_name:str, api_key=get_api_key("groq"), temperature=DEFAULT_TEMPERATURE):
    return ChatGroq(model_name=model_name, temperature=temperature, api_key=api_key) # type: ignore
   
# OpenRouter models
def get_openrouter_chat(model_name: str, api_key=get_api_key("openrouter"), temperature=DEFAULT_TEMPERATURE, base_url=os.getenv("OPEN_ROUTER_BASE_URL") or "https://openrouter.ai/api/v1"):
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, base_url=base_url) # type: ignore
      
def get_openrouter_embedding(model_name: str, api_key=get_api_key("openrouter"), base_url=os.getenv("OPEN_ROUTER_BASE_URL") or "https://openrouter.ai/api/v1"):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url) # type: ignore

# Sambanova models
def get_sambanova_chat(model_name: str, api_key=get_api_key("sambanova"), temperature=DEFAULT_TEMPERATURE, base_url=os.getenv("SAMBANOVA_BASE_URL") or "https://fast-api.snova.ai/v1", max_tokens=1024):
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, base_url=base_url, max_tokens=max_tokens) # type: ignore

# Deepseek models
def get_deepseek_chat(model_name:str="deepseek-chat", api_key=get_api_key("deepseek"), temperature=DEFAULT_TEMPERATURE, base_url="https://api.deepseek.com/v1"):
    return ChatOpenAI(
        model_name=model_name, 
        temperature=temperature, 
        api_key=api_key, 
        base_url=base_url
    )

def get_deepseek_embedding(model_name:str="deepseek-text-embedding", api_key=get_api_key("deepseek"), base_url="https://api.deepseek.com/v1"):
    return OpenAIEmbeddings(
        model=model_name, 
        api_key=api_key, 
        base_url=base_url
    )

# Zhipu AI models
def get_zhipu_chat(model_name:str="glm-4", api_key=get_api_key("zhipu"), temperature=DEFAULT_TEMPERATURE):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)
    return client.chat.completions.create(
        model=model_name,
        temperature=temperature
    )

def get_zhipu_embedding(model_name:str="embedding-2", api_key=get_api_key("zhipu")):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)
    return client.embeddings.create(
        model=model_name
    )

def get_glm4v_chat(image_path: str, prompt: str, api_key=get_api_key("zhipu")):
    import base64
    from zhipuai import ZhipuAI
    
    with open(image_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    
    client = ZhipuAI(api_key=api_key)
    return client.chat.completions.create(
        model="glm-4v-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

# Zhipu Web-Search-Pro
def get_websearch_pro(query: str, api_key=get_api_key("zhipu")):
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": "web-search-pro",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        url,
        json=data,
        headers={'Authorization': api_key},
        timeout=300
    )
    return response.json()
