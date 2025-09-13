import sys
import os

# Add the compiled Pybind11 module path (lib folder contains utils .so file)
sys.path.append(os.path.join(os.path.dirname(__file__), "../lib"))

# Import the compiled C++ module
import quantum_classical_hybrid as qc

# Load environment variables via your C++ dotenv bindings
env_path = os.path.join(os.path.dirname(__file__), "../.env")

class PreprocessingPipeline:
    def __init__(self):
        # Initialize preprocessing classes from the C++ bindings
        self.bert_normaliser = qc.BertNormaliser()
        self.byte_normaliser = qc.ByteNormalizer()
        self.digit_normaliser = qc.DigitNormaliser()
        self.metaspace_normaliser = qc.MetaspaceNormaliser()
        self.prepend = None  # Will initialize after normalization
        self.tokenizer = qc.Tokenizer()

    def run(self, text):
        # --- Bert normalization ---
        normalized_text = self.bert_normaliser.bertCleaning(text)
        normalized_text = self.bert_normaliser.stripAccents(normalized_text)

        # --- Byte normalization ---
        byte_tokens = self.byte_normaliser.ByteNormalise(normalized_text, True)

        # --- Digit normalization ---
        try:
            digit_tokens = self.digit_normaliser.normaliseDigits(normalized_text, True)
        except Exception as e:
            print(f"DigitNormaliser failed: {e}")
            digit_tokens = [normalized_text]

        # Join digit tokens for metaspace
        digit_text = "".join(digit_tokens)

        # --- Metaspace normalization ---
        try:
            self.metaspace_normaliser.setReplacement(' ', True)
            metaspace_tokens = self.metaspace_normaliser.pretok(digit_text, True)
        except Exception as e:
            print(f"MetaspaceNormaliser failed: {e}")
            metaspace_tokens = digit_tokens

        metaspace_text = "".join(metaspace_tokens)

        # --- Prepend normalization ---
        self.prepend = qc.Prepend("dummy.txt", metaspace_text)
        normalized_values = set(self.prepend.extract_normalised(metaspace_text))

        # --- Tokenization ---
        tokens = self.tokenizer.tokenize(normalized_text)
        stats = {
            "total_tokens": self.tokenizer.countTokens(tokens),
            "unique_tokens": self.tokenizer.countUniqueTokens(tokens),
            "total_words": self.tokenizer.countWords(tokens),
            "total_punctuation": self.tokenizer.countPunctuation(normalized_text),
            "sentences": self.tokenizer.countSentences(normalized_text)
        }

        return {
            "normalized_text": normalized_text,
            "byte_tokens": byte_tokens,
            "digit_tokens": digit_tokens,
            "metaspace_tokens": metaspace_tokens,
            "stats": stats
        }

# Example usage
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    text = "This is a test input 123!"
    result = pipeline.run(text)
    print(result)