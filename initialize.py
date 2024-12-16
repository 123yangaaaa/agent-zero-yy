import os
import models
from agent import AgentConfig
from python.helpers import files

def initialize():
    
    # Get model configuration from environment variables
    model_type = os.getenv("MODEL_TYPE", "deepseek-chat")  # 默认使用 deepseek-chat
    image_path = os.getenv("IMAGE_PATH", "")  # 用于 GLM-4V 的图片路径
    image_prompt = os.getenv("IMAGE_PROMPT", "")  # 用于 GLM-4V 的提示词

    # Select chat model based on environment variable
    if model_type == "deepseek-chat":
        chat_llm = models.get_deepseek_chat(model_name="deepseek-chat-v1-32k", temperature=0)
    elif model_type == "deepseek-coder":
        chat_llm = models.get_deepseek_chat(model_name="deepseek-coder-v1-32k", temperature=0)
    elif model_type == "glm4v" and image_path and image_prompt:
        chat_llm = models.get_glm4v_chat(image_path=image_path, prompt=image_prompt)
    else:
        # 默认fallback到 deepseek-chat
        chat_llm = models.get_deepseek_chat(model_name="deepseek-chat-v1-32k", temperature=0)

    # utility model used for helper functions (cheaper, faster)
    utility_llm = models.get_deepseek_chat(model_name="deepseek-chat-v1-32k", temperature=0)

    # embedding model used for memory
    embedding_llm = models.get_deepseek_embedding(model_name="text-embedding-v1")

    # agent configuration
    config = AgentConfig(
        chat_model = chat_llm,
        utility_model = utility_llm,
        embeddings_model = embedding_llm,
        knowledge_subdirs = ["default","custom"],
        auto_memory_count = 0,
        rate_limit_requests = 30,
        max_tool_response_length = 3000,
        code_exec_docker_enabled = True,
        code_exec_ssh_enabled = True,
        additional = {
            "model_type": model_type,  # 记录当前使用的模型类型
            "glm4v_enabled": model_type == "glm4v",  # 是否启用了 GLM-4V
        },
    )

    # return config object
    return config
