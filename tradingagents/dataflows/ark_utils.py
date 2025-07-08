import json
import time

import requests


def get_web_search_result(prompt):
    url = 'https://api.coze.cn/v3/chat'
    headers = {
        'Authorization': 'Bearer pat_xxxx',
        'Content-Type': 'application/json'
    }
    payload = {
        "bot_id": "752xxxx",
        "user_id": "123123",
        "stream": False,
        "auto_save_history": True,
        "additional_messages": [
            {
                "role": "user",
                "content": prompt,
                "content_type": "text"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 检查请求是否成功
        print("请求成功")
        print(response.json())  # 打印JSON格式的响应
        # {
        #     "data": {
        #         "id": "7524607829328560178", chat_id
        #         "conversation_id": "7524607829328543794",
        #         "bot_id": "7524548609812103222",
        #         "created_at": 1751959284,
        #         "last_error": {
        #             "code": 0,
        #             "msg": ""
        #         },
        #         "status": "in_progress"
        #     },
        #     "code": 0,
        #     "msg": ""
        # }
        if response.json()["data"]["status"] == "in_progress":
            # 开始轮询 每2秒一次 一直到状态变为 completed
            chat_id = response.json()["data"]["id"]
            conversation_id = response.json()["data"]["conversation_id"]
            url = f'https://api.coze.cn/v3/chat/retrieve?conversation_id={conversation_id}&chat_id={chat_id}'
            while True:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                if response.json()["data"]["status"] == "completed":
                    break
                time.sleep(2)
            url = f'https://api.coze.cn/v3/chat/message/list?conversation_id={conversation_id}&chat_id={chat_id}&'
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()["data"]
            # print(response.json()["data"])
            result = result[1]['content'] + '\n' + result[2]['content']
            time.sleep(60)
            return result
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"请求异常: {req_err}")
    except json.JSONDecodeError:
        print("响应内容不是有效的JSON格式")

if __name__ == "__main__":
    result = get_web_search_result('请帮我查询舍得酒业最近的新闻')
    print(result)