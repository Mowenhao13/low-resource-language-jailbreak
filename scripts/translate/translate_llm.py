from openai import OpenAI
import time

class LLMTranslator:
    def __init__(self, model_id: str = "qwen3-instruct",
                 base_url: str = "https://aigw.sysu.edu.cn/v1",
                 api_key: str = "sk-PKikgGT1Mi4VJLeO9u0m27Z26UvReC5Gb89W3UENLd1Eg5mx"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_id = model_id
        self.chunk_size = 2000  # Safe chunk size for LLM context

    def Translate(self, source_text: str, source_lang: str, target_lang: str) -> str:
        if not source_text or len(source_text.strip()) == 0:
            return source_text

        # If text is short enough, translate directly
        if len(source_text) <= self.chunk_size:
            return self._translate_chunk(source_text, source_lang, target_lang)

        # Otherwise, split into chunks and translate
        chunks = self._split_text(source_text)
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                translated = self._translate_chunk(chunk, source_lang, target_lang)
                translated_chunks.append(translated)
                if i < len(chunks) - 1:
                    time.sleep(0.5)  # Avoid rate limiting

        return ''.join(translated_chunks)

    def _split_text(self, text: str) -> list:
        """Split text into safe chunks, trying to split at natural boundaries."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to find a natural break point
            if end < text_length:
                # Look for paragraph break
                break_pos = text.rfind('\n\n', start, end)
                if break_pos == -1:
                    # Look for line break
                    break_pos = text.rfind('\n', start, end)
                if break_pos == -1:
                    # Look for period followed by space
                    break_pos = text.rfind('. ', start, end)
                if break_pos != -1:
                    end = break_pos + 1

            chunks.append(text[start:end])
            start = end

        return chunks

    def _translate_chunk(self, source_text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk of text using LLM."""
        system_prompt = f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Only output the translation, do not include any explanations or notes."

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_text}
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM Translation warning: {e}")
            # Return original text if translation fails
            return source_text


def main():
    # Test the translator
    translator = LLMTranslator()

    # Simple test
    source_text = "Hello world! This is a test of the LLM translator."
    source_lang = "en"
    target_lang = "zh"

    print(f"Source: {source_text}")
    result = translator.Translate(source_text, source_lang, target_lang)
    print(f"Translation: {result}")


if __name__ == "__main__":
    main()
