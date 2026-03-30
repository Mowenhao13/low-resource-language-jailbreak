import requests
from urllib.parse import quote

url = f'https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=en&tl=gd&q={
    quote("你好啊，亲爱的朋友.今天天气不错哦。我有肉吃，你有吗")
}'

# 发送 GET 请求
res = requests.get(url)

# 提取翻译结果
text = [te[0] for te in res.json()[0]]
print(text)
