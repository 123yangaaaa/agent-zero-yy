import models
import uuid
import requests

def perplexity_search(query:str, api_key=None):
    """
    使用智谱 Web-Search-Pro 进行搜索，替代原有的 Perplexity 搜索
    """
    api_key = api_key or models.get_api_key("zhipu")
    
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
    
    try:
        response = requests.post(
            url,
            json=data,
            headers={'Authorization': api_key},
            timeout=300
        )
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Web-Search-Pro search failed: {str(e)}")
    return ""