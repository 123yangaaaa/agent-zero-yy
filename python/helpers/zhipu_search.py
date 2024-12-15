import models

def zhipu_search(query: str, api_key=None):
    api_key = api_key or models.get_api_key("zhipu")
    response = models.get_websearch_pro(query, api_key)
    
    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["message"]["content"]
    return ""
