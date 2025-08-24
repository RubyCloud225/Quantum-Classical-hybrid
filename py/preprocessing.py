import preprocessing
# import ModelCircuit  # Uncomment this line if ModelCircuit is available and required

class PreprocessingPipeline:
    def __init__(self):
        # Initialize all normalizers and tokenizers
        self.bert = preprocessing.BertNormaliser()
        self.byte_norm = preprocessing.ByteNormalizer()
        self.digit_norm = preprocessing.DigitNormaliser()
        self.metaspace = preprocessing.MetaspaceNormaliser()
        self.tokenizer = preprocessing.Tokeniser()
        self.img_tokenizer = preprocessing.ImageSequentialTokenizer()
        #configure metaspace replacement
        self.metaspace.setReplacement("Ġ", True)
        self.metaspace.setReplacement(" ", True)
    
    def preprocess_text(self, text_input):
        #BERT Cleaning
        cleaned_text = self.bert.bertCleaning(text_input)
        cleaned_text = self.bert.stripAccents(cleaned_text)
        #Byte Normalisation
        byte_tokens = self.byte_norm.ByteNormalise(cleaned_text, True)
        # Digit normalization
        digit_normalized = self.digit_norm.normaliseDigits(byte_tokens, True)
        # Metaspace normalisation
        metaspace_text = self.metaspace.pretok(digit_normalized, True)
        #Tokenizer
        tokens = self.tokenizer.tokenize(metaspace_text)
        return {
            "cleaned_text": cleaned_text,
            "byte_tokens": byte_tokens,
            "metaspace_text": metaspace_text,
            "tokens": tokens,
            "total_tokens": self.tokenizer.countTokens(tokens),
            "unique_tokens": self.tokenizer.countUniqueTokens(tokens),
            "words": self.tokenizer.countWords(tokens),
            "punctuation": self.tokenizer.countPunctuation(metaspace_text),
        }
    def preprocess_image(self, image_path):
        # Sequential Patch tokenization
        sample_data = self.img_tokenizer.tokenizeImage(image_path)
        return sample_data
    
    def merge_text_image(self, text_input, image_path):
        """
        Returns SampleData object with both text tokens and image sequential tokens.
        """
        text_result = self.preprocess_text(text_input)
        image_sample = self.preprocess_image(image_path)
        # create new SampleData Object
        merged_sample = preprocessing.SampleData()
        merged_sample.textTokens = text_result["tokens"]
        merged_sample.imageTokens = image_sample.imageTokens

        # Optionally copy over other SampleData fields if needed
        merged_sample.token_embedding = [
            text_result["total_tokens"],
            text_result["unique_tokens"],
            text_result["words"],
            text_result["punctuation"]
        ]
        return merged_sample

class SampleDataWrapper:
    def __init__(self, sample_data):
        self._sample_data = sample_data
    
    @property
    def noise(self):
        return self._sample_data.noise
    @property
    def normalized_noise(self):
        return self._sample_data.normalized_noise
    @property
    def target_values(self):
        return self._sample_data.target_value
    @property
    def density(self):
        return self._sample_data.density
    @property
    def nll(self):
        return self._sample_data.nll
    @property
    def entropy(self):
        return self._sample_data.entropy
    @property
    def GaussianNoise(self):
        return self._sample_data.GaussianNoise
    @property
    def LayereNormalization(self):
        return self._sample_data.LayerNormalization
    @property
    def LinearRegression(self):
        return self._sample_data.LinearRegression

# example usage
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()

    text_input = "The price is 1,234.56 USD on 23rd March 2025."
    image_path = "cat.png"

    merged_sample = pipeline.merge_text_image(text_input, image_path)
    wrapped = SampleDataWrapper(merged_sample)
    print("Text tokens:", merged_sample.textTokens)
    print("Number of image patches:", len(merged_sample.imageTokens))
    if wrapped.sample_data.imageTokens:
        first_patch = merged_sample.imageTokens[0]
        print("First patch embedding length:", len(first_patch.embedding))
        print("First patch position (row, col):", first_patch.row, first_patch.col)
    print("\n==== Sample Data Summary ====")
    print("Noise:", wrapped.noise)
    print("Normalized Noise:", wrapped.normalized_noise)
    print("Target Values:", wrapped.target_values)
    print("Density:", wrapped.density)
    print("NLL:", wrapped.nll)
    print("Entropy:", wrapped.entropy)