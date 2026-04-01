"""
NLLB 翻译器 - 持久化模型加载版本

支持多语言翻译，模型只需加载一次，后续调用无需重新加载。
使用 Facebook 的 NLLB-200 模型，支持 200 多种语言。
"""

import os
import warnings
import torch
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 忽略 CUDA 和弃用警告
warnings.filterwarnings("ignore", message="CUDA initialization")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")


class NLLBTranslator:
    """
    NLLB 翻译器类 - 模型持久化加载

    使用示例:
        translator = NLLBTranslator(model_dir="~/data/models/nllb-200")
        result = translator.translate("Hello, world!", source_lang="eng_Latn", target_lang="zho_Hans")
    """

    # 类级别的缓存，所有实例共享同一个模型（如果使用相同的模型目录）
    _model_cache: dict = {}
    _tokenizer_cache: dict = {}

    def __init__(
        self,
        model_dir: str = "~/data/models/nllb-200",
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        初始化 NLLB 翻译器

        Args:
            model_dir: 本地模型存储目录
            model_name: Hugging Face 模型名称（首次下载时使用）
            device: 使用的设备，如 "cuda" 或 "cpu"，默认自动选择
            use_cache: 是否使用类级别的模型缓存，避免重复加载
        """
        self.model_dir = os.path.expanduser(model_dir)
        self.model_name = model_name
        self.use_cache = use_cache

        # 自动选择设备 - 优先使用 GPU
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[NLLB] 检测到 CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print(f"[NLLB] 警告: CUDA 不可用，使用 CPU (速度较慢)")
        else:
            self.device = device
            print(f"[NLLB] 使用指定设备: {device}")

        # 确保目录存在
        os.makedirs(self.model_dir, exist_ok=True)

        # 加载模型和分词器
        self._load_model()

    def _load_model(self):
        """加载模型和分词器，支持缓存机制"""
        cache_key = self.model_dir

        # 检查是否使用缓存且缓存中已有模型
        if self.use_cache and cache_key in NLLBTranslator._model_cache:
            print(f"[NLLB] 使用缓存的模型: {cache_key}")
            self.model = NLLBTranslator._model_cache[cache_key]
            self.tokenizer = NLLBTranslator._tokenizer_cache[cache_key]
            return

        # 检查本地是否已有模型
        if os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"[NLLB] 从本地加载模型: {self.model_dir}")
            model_path = self.model_dir
        else:
            print(f"[NLLB] 下载模型: {self.model_name}")
            print(f"[NLLB] 模型将保存到: {self.model_dir}")
            model_path = self.model_name

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.model_dir,
            local_files_only=os.path.exists(os.path.join(self.model_dir, "config.json"))
        )

        # 加载模型 - 简化方式，避免 device_map 问题
        if self.device == "cuda":
            # GPU 模式：使用 float16 加速
            print(f"[NLLB] 以 float16 精度加载模型到 GPU...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir=self.model_dir,
                local_files_only=os.path.exists(os.path.join(self.model_dir, "config.json")),
                torch_dtype=torch.float16
            )
            self.model = self.model.to(self.device)
        else:
            # CPU 模式
            print(f"[NLLB] 以 float32 精度加载模型到 CPU...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir=self.model_dir,
                local_files_only=os.path.exists(os.path.join(self.model_dir, "config.json")),
                torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)

        # 如果是首次加载且本地没有，保存模型
        if not os.path.exists(os.path.join(self.model_dir, "config.json")):
            print(f"[NLLB] 保存模型到本地: {self.model_dir}")
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)

        # 保存到缓存
        if self.use_cache:
            NLLBTranslator._model_cache[cache_key] = self.model
            NLLBTranslator._tokenizer_cache[cache_key] = self.tokenizer

        print(f"[NLLB] 模型加载完成，设备: {self.device}")

    def _get_lang_token_id(self, lang_code: str) -> Optional[int]:
        """
        获取语言代码对应的 token ID，兼容不同版本的 tokenizer

        Args:
            lang_code: 语言代码（如 "zho_Hans"）

        Returns:
            token ID 或 None（如果无法获取）
        """
        # 方法 1: 尝试直接访问 lang_code_to_id (旧版 API)
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            if lang_code in self.tokenizer.lang_code_to_id:
                return self.tokenizer.lang_code_to_id[lang_code]

        # 方法 2: 使用 convert_tokens_to_ids (新版 TokenizersBackend)
        try:
            # 直接尝试语言代码
            token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return token_id
        except Exception:
            pass

        # 方法 3: 尝试添加特殊 token 标记
        try:
            # 某些 tokenizer 需要特殊格式
            for fmt in [f"__{lang_code}__", f"<{lang_code}>", f"[{lang_code}]"]:
                token_id = self.tokenizer.convert_tokens_to_ids(fmt)
                if token_id is not None and token_id != self.tokenizer.unk_token_id:
                    return token_id
        except Exception:
            pass

        print(f"[NLLB] 警告: 无法获取语言代码 {lang_code} 的 token ID")
        return None

    def translate(
        self,
        text: str,
        source_lang: str = "eng_Latn",
        target_lang: str = "zho_Hans",
        max_length: int = 512,
        enable_segment: bool = True,
        segment_threshold: int = 500,
        **generate_kwargs
    ) -> str:
        """
        翻译文本

        Args:
            text: 要翻译的文本
            source_lang: 源语言代码（如 "eng_Latn"）
            target_lang: 目标语言代码（如 "zho_Hans"）
            max_length: 最大生成长度
            enable_segment: 是否对长文本启用分段翻译
            segment_threshold: 分段阈值（字符数）
            **generate_kwargs: 其他生成参数

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return text

        # 如果文本很长，分段翻译
        if enable_segment and len(text) > segment_threshold:
            return self._translate_segmented(
                text, source_lang, target_lang, max_length, **generate_kwargs
            )

        # 单段翻译
        return self._translate_single(
            text, source_lang, target_lang, max_length, **generate_kwargs
        )

    def _translate_single(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int,
        **generate_kwargs
    ) -> str:
        """单段翻译的内部实现"""
        # 设置源语言 - 兼容不同版本的 tokenizer
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_lang

        # 获取目标语言的 BOS token ID
        forced_bos_token_id = self._get_lang_token_id(target_lang)

        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # 生成翻译 - 添加默认控制参数防止循环生成
        generate_kwargs_copy = generate_kwargs.copy()
        if forced_bos_token_id is not None:
            generate_kwargs_copy['forced_bos_token_id'] = forced_bos_token_id

        # 添加防止循环生成的默认参数
        if 'no_repeat_ngram_size' not in generate_kwargs_copy:
            generate_kwargs_copy['no_repeat_ngram_size'] = 3
        if 'repetition_penalty' not in generate_kwargs_copy:
            generate_kwargs_copy['repetition_penalty'] = 1.2

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                max_length=max_length,
                **generate_kwargs_copy
            )

        # 解码
        translated_text = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        # 清理可能的问题字符
        translated_text = self._clean_translation_output(translated_text)

        return translated_text

    def _translate_segmented(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int,
        **generate_kwargs
    ) -> str:
        """分段翻译：按行分割长文本，逐段翻译后合并"""
        # 按行分割
        lines = text.split('\n')
        translated_lines = []

        for line in lines:
            if not line.strip():
                translated_lines.append(line)
                continue

            # 单行也很长的话，按句子进一步分割
            if len(line) > 300:
                # 按标点符号分割
                import re
                segments = re.split(r'([。！？.!?]+)', line)
                current_segment = ""
                for i in range(0, len(segments), 2):
                    part = segments[i]
                    punct = segments[i+1] if i+1 < len(segments) else ""

                    if len(current_segment + part + punct) < 300:
                        current_segment += part + punct
                    else:
                        if current_segment:
                            translated = self._translate_single(
                                current_segment, source_lang, target_lang, max_length, **generate_kwargs
                            )
                            translated_lines.append(translated)
                        current_segment = part + punct

                if current_segment:
                    translated = self._translate_single(
                        current_segment, source_lang, target_lang, max_length, **generate_kwargs
                    )
                    translated_lines.append(translated)
            else:
                translated = self._translate_single(
                    line, source_lang, target_lang, max_length, **generate_kwargs
                )
                translated_lines.append(translated)

        return '\n'.join(translated_lines)

    def _clean_translation_output(self, text: str) -> str:
        """清理翻译输出中的问题字符"""
        # 移除重复的无限符号
        if '∞' in text:
            # 如果有大量 ∞，只保留前面有意义的部分
            parts = text.split('∞', 1)
            if len(parts[0].strip()) > 10:
                text = parts[0].rstrip()
            else:
                # 如果前面内容太短，尝试找到第一个有意义的位置
                import re
                meaningful_match = re.search(r'[a-zA-Z\u0080-\uFFFF]{10,}', text)
                if meaningful_match:
                    text = text[:meaningful_match.end()]
                else:
                    text = text.split('∞')[0]

        return text

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "eng_Latn",
        target_lang: str = "zho_Hans",
        max_length: int = 512,
        batch_size: int = 8,
        **generate_kwargs
    ) -> List[str]:
        """
        批量翻译文本

        Args:
            texts: 要翻译的文本列表
            source_lang: 源语言代码
            target_lang: 目标语言代码
            max_length: 最大生成长度
            batch_size: 批次大小
            **generate_kwargs: 其他生成参数

        Returns:
            翻译后的文本列表
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._translate_batch_impl(
                batch, source_lang, target_lang, max_length, **generate_kwargs
            )
            results.extend(batch_results)

        return results

    def _translate_batch_impl(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int,
        **generate_kwargs
    ) -> List[str]:
        """批量翻译的内部实现"""
        # 设置源语言 - 兼容不同版本的 tokenizer
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_lang

        # 获取目标语言的 BOS token ID
        forced_bos_token_id = self._get_lang_token_id(target_lang)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        generate_kwargs_copy = generate_kwargs.copy()
        if forced_bos_token_id is not None:
            generate_kwargs_copy['forced_bos_token_id'] = forced_bos_token_id

        # 添加防止循环生成的默认参数
        if 'no_repeat_ngram_size' not in generate_kwargs_copy:
            generate_kwargs_copy['no_repeat_ngram_size'] = 3
        if 'repetition_penalty' not in generate_kwargs_copy:
            generate_kwargs_copy['repetition_penalty'] = 1.2

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                max_length=max_length,
                **generate_kwargs_copy
            )

        translated_texts = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )

        # 清理每个翻译结果
        translated_texts = [self._clean_translation_output(t) for t in translated_texts]

        return translated_texts

    @staticmethod
    def clear_cache():
        """清除模型缓存，释放内存"""
        NLLBTranslator._model_cache.clear()
        NLLBTranslator._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[NLLB] 缓存已清除")


# 语言代码映射表（常用语言）
LANG_CODES = {
    # 中文
    "zh": "zho_Hans",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    # 英语
    "en": "eng_Latn",
    "en-US": "eng_Latn",
    # 西班牙语
    "es": "spa_Latn",
    # 法语
    "fr": "fra_Latn",
    # 德语
    "de": "deu_Latn",
    # 日语
    "ja": "jpn_Jpan",
    # 韩语
    "ko": "kor_Hang",
    # 俄语
    "ru": "rus_Cyrl",
    # 阿拉伯语
    "ar": "arb_Arab",
    # 葡萄牙语
    "pt": "por_Latn",
    # 意大利语
    "it": "ita_Latn",
    # 印地语
    "hi": "hin_Deva",
    # 孟加拉语
    "bn": "ben_Beng",
    # 越南语
    "vi": "vie_Latn",
    # 泰语
    "th": "tha_Thai",
    # 马来语
    "ms": "msa_Latn",
    # 印度尼西亚语
    "id": "ind_Latn",
    # 荷兰语
    "nl": "nld_Latn",
    # 波兰语
    "pl": "pol_Latn",
    # 土耳其语
    "tr": "tur_Latn",
    # 乌克兰语
    "uk": "ukr_Cyrl",
    # 希腊语
    "el": "ell_Grek",
    # 希伯来语
    "he": "heb_Hebr",
    # 斯瓦希里语
    "sw": "swh_Latn",
    # 祖鲁语 (Zulu)
    "zu": "zul_Latn",
    # 苏格兰盖尔语 (Scottish Gaelic)
    "gd": "gla_Latn",
    # 苗语 (Hmong) - NLLB distilled 模型不支持，用科萨语 (Xhosa) 替代
    "hmn": "xho_Latn",
    # 科萨语 (Xhosa) - 替代苗语的低资源语言
    "xh": "xho_Latn",
    # 南索托语 (Sotho) - 另一个低资源语言选项
    "st": "sot_Latn",
    # 瓜拉尼语 (Guarani)
    "gn": "grn_Latn",
}


def get_nllb_lang_code(lang: str) -> str:
    """
    获取 NLLB 语言代码

    Args:
        lang: 简单语言代码（如 "zh", "en"）或完整的 NLLB 代码

    Returns:
        NLLB 语言代码
    """
    if lang in LANG_CODES:
        return LANG_CODES[lang]
    # 如果已经是完整的 NLLB 代码，直接返回
    return lang


def main():
    """简单测试"""
    print("=" * 60)
    print("NLLB 翻译器测试")
    print("=" * 60)

    # 初始化翻译器
    translator = NLLBTranslator(
        model_dir="~/data/models/nllb-200",
        model_name="facebook/nllb-200-distilled-600M"
    )

    print("\n" + "-" * 60)
    print("测试 1: 英文 -> 中文")
    print("-" * 60)

    text = "Hello, world! This is a test of the NLLB translator."
    print(f"原文: {text}")

    result = translator.translate(
        text,
        source_lang="eng_Latn",
        target_lang="zho_Hans"
    )
    print(f"翻译: {result}")

    print("\n" + "-" * 60)
    print("测试 2: 使用简写语言代码")
    print("-" * 60)

    result = translator.translate(
        text,
        source_lang=get_nllb_lang_code("en"),
        target_lang=get_nllb_lang_code("zh")
    )
    print(f"翻译: {result}")

    print("\n" + "-" * 60)
    print("测试 3: 批量翻译")
    print("-" * 60)

    texts = [
        "How are you?",
        "What is your name?",
        "Nice to meet you!"
    ]

    results = translator.translate_batch(
        texts,
        source_lang=get_nllb_lang_code("en"),
        target_lang=get_nllb_lang_code("zh")
    )

    for src, tgt in zip(texts, results):
        print(f"{src:30} -> {tgt}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
