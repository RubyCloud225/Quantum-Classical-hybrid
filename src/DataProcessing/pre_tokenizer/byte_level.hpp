// convert bytes to unicode characters
// containing all values some are 2_8 + n where n is 0 to 255
// see https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs

// turn values into UTF 32

// regex that matches exactly one token

//steps to handle tokenization 
// 1. convert UTF8 to string
// 2. add leading space to the first word
// 3. trim offsets to void whitespaces
// 4. use regex to match tokens