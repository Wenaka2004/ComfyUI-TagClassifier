import folder_paths
import json
import requests
import torch
import re

# 从文件中读取历史记录
def load_prompt_history(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 调用DeepSeek API
def call_deepseek_api(api_key, user_input, prompt_history):
    # 添加用户输入到历史记录
    prompt_history['messages'].append({"role": "user", "content": user_input})
    
    # 设置API请求的URL和headers
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 设置请求体
    data = {
        "model": "deepseek-chat",
        "messages": prompt_history['messages']
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    
    # 检查响应状态
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

# 提取 JSON 部分
def extract_json_from_markdown(markdown_text):
    # 使用正则表达式提取 ```json 和 ``` 之间的内容
    match = re.search(r'```json\n(.*?)\n```', markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回 JSON 部分
    else:
        raise ValueError("No JSON content found in the markdown text.")

class LLMProcessingNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IS_NSFW", "角色头部以上服饰特征", "角色动作及表情", "角色上半身服饰特征", "角色下半身服饰特征", "其他", "NSFW")
    FUNCTION = "process"
    CATEGORY = "LLM Tag Classifier"

    def process(self, text, api_key):
        # 从文件中加载历史记录
        prompt_history = load_prompt_history("custom_nodes\Comfyui-TagClassifier\prompt.json")
        
        # 调用DeepSeek API
        try:
            response = call_deepseek_api(api_key, text, prompt_history)
            llm_output = response['choices'][0]['message']['content']
            
            # 提取 JSON 部分
            json_content = extract_json_from_markdown(llm_output)
            
            # 打印提取的 JSON 内容，用于调试
            print("Extracted JSON Content:", json_content)
            
            # 解析 JSON
            llm_output_json = json.loads(json_content)
            
            # 打印解析后的 JSON 内容，用于调试
            print("Parsed JSON Content:", llm_output_json)
            
            # 提取七个内容
            is_nsfw = llm_output_json.get("IS_NSFW", "")
            head_features = llm_output_json.get("\u89d2\u8272\u5934\u90e8\u4ee5\u4e0a\u670d\u9970\u7279\u5f81", "")
            action_expression = llm_output_json.get("\u89d2\u8272\u52a8\u4f5c\u53ca\u8868\u60c5", "")
            upper_body_features = llm_output_json.get("\u89d2\u8272\u4e0a\u534a\u8eab\u670d\u9970\u7279\u5f81", "")
            lower_body_features = llm_output_json.get("\u89d2\u8272\u4e0b\u534a\u8eab\u670d\u9970\u7279\u5f81", "")
            other = llm_output_json.get("\u5176\u4ed6", "")
            nsfw = llm_output_json.get("NSFW", "")
            
            return (is_nsfw, head_features, action_expression, upper_body_features, lower_body_features, other, nsfw)
        except Exception as e:
            raise Exception(f"Error processing LLM output: {e}")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LLMProcessingNode": LLMProcessingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMProcessingNode": "LLM Tag Classifier",
}