from datasets import load_dataset
import preprocessing

"""
using MS Coco Standard Captions dataset
a separate dataset will be used for digits and other languages.

"""
def batch_process_tokenise(dataset, batchsize=32, tokenizer=preprocessing.tokenizer):
    """
    Uses custom tokenizer to process dataset in batches.
    Args:
        dataset: Dataset to be processed.
        batchsize: Size of each batch. - size of the batch initially is 32
        tokenizer: Tokenizer to use for processing.
    Returns:
        List of processed batches.
    """
    total= len(dataset)
    for i in range(0, total, batchsize):
        batch = dataset.select(range(i, min(i + batchsize, total)))
        processed_batch = tokenizer(batch['caption'], truncation=True, padding='max_length', max_length=128)
        tokenised_dataset = {
            'input_ids': processed_batch['input_ids'],
            'attention_mask': processed_batch['attention_mask'],
            'labels': processed_batch['input_ids']
        }
        for example in batch:
            caption = example['caption']
            # run the preprocessing pipeline
            preprocessed_caption = preprocessing.run_preprocessing(caption)
            

def process_data_captions():
    #Load MS-Coco Standard Captions dataset
    dataset = load_dataset("coco_captions")
    # spilt between test and train
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    print(f"train datset: ${len(train_dataset)}")
    print(f"test datset: ${len(test_dataset)}")


