from site import makepath
import preprocessing
import utils
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
    
    def nasa_preprocess_text(self, text_input):
        env_vars = utils.load_env(".env")
        nasa_api_key = env_vars.get("NASA_API_KEY")
        if not nasa_api_key:
            raise ValueError("NASA_API_KEY not found in environment variables.")
        response = env_vars.get(nasa_api_key, text_input)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch NASA APOD data: {response.status_code}")
        return response.text
    
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
    def LayerNormalization(self):
        return self._sample_data.LayerNormalization
    @property
    def LinearRegression(self):
        return self._sample_data.LinearRegression

class SanityMetrics:
    def __init__(self, sample_data=None, text_input=None, image_path=None):
        self.sample_data = sample_data
        self.text_input = text_input
        self.image_path = image_path
        self.pipeline = PreprocessingPipeline()
        if self.sample_data is None and self.text_input and self.image_path:
            self.sample_data = self.pipeline.merge_text_image(self.text_input, self.image_path)
        self.metrics = {}
    
    def _compute_from_sample_data(self):
        text_tokens = getattr(self.sample_data, 'textTokens', [])
        num_text_tokens = len(text_tokens)
        unique_tokens = len(set(text_tokens))
        text_length = sum(len(t) for t in text_tokens) if num_text_tokens > 0 else 0
        avg_token_length = text_length / num_text_tokens if num_text_tokens > 0 else 0
        image_tokens = getattr(self.sample_data, 'imageTokens', [])
        num_image_patches = len(image_tokens)
        avg_embedding_length = (sum(len(patch.embedding) for patch in image_tokens) / num_image_patches) if num_image_patches > 0 else 0
        self.metrics = {
            "num_text_tokens": num_text_tokens,
            "unique_text_tokens": unique_tokens,
            "avg_token_length": avg_token_length,
            "num_image_patches": num_image_patches,
            "avg_embedding_length": avg_embedding_length
        }
    def _compute_from_raw_inputs(self):
        #use pipeline to preprocess
        text_metrics = self.pipeline.sanity_check_text(self.text_input)
        image_metrics = self.pipeline.sanity_check_image(self.image_path)
        self.metrics = {"text_metrics": text_metrics, "image_metrics": image_metrics}
    def get_metrics(self):
        if self.sample_data:
            self._compute_from_sample_data()
        elif self.text_input and self.image_path:
            self._compute_from_raw_inputs()
        else:
            raise ValueError("Insufficient data to compute metrics.")
        return self.metrics

    


# example usage
if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    sample_text = "Sample text for preprocessing."
    text_input_dict = pipeline.nasa_preprocess_text(sample_text)
    sample_image = "path/to/image.jpg"  # Replace with actual image path
    merged_sample = pipeline.merge_text_image(sample_text, sample_image)
    wrapped = SampleDataWrapper(merged_sample)
    print("Text tokens:", merged_sample.textTokens)
    print("Number of image patches:", len(merged_sample.imageTokens))
    if merged_sample.imageTokens:
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
    print("Gaussian Noise:", wrapped.GaussianNoise)
    print("Layer Normalization:", wrapped.LayerNormalization)
    print("Linear Regression:", wrapped.LinearRegression)
    # Compute sanity metrics
    sanity = SanityMetrics(sample_data=merged_sample)
    metrics = sanity.get_metrics()
    print("Sanity Metrics:", metrics)
    sanity_raw = SanityMetrics(text_input=sample_text, image_path=sample_image)
    raw_metrics = sanity_raw.get_metrics()
    print("Raw Input Sanity Metrics:", raw_metrics)
