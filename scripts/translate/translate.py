import requests
from urllib.parse import quote

class Translator:
    def __init__(self):
        pass

    def Translate(self, source_text: str, source_lang: str, target_lang: str) -> str:
        url = f'https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl={source_lang}&tl={target_lang}&q={
            quote(source_text)
        }'
        res = requests.get(url)
        # 提取翻译结果
        data = res.json()
    
    # 方法1: 直接拼接所有翻译片段
        text = ''.join([te[0] for te in data[0] if te[0]])  
        return text 
    
def main():
    translator = Translator() 
    source_text = "Hello world"
    source_lang = "en"
    target_lang = "zh-CN"
    print(translator.Translate(source_text, source_lang, target_lang))

if __name__ == "__main__":
    main()