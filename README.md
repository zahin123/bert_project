 steps to run :  
 cd ~/Desktop/bert_project
pip3 install transformers datasets accelerate
python3 baseline_mlm.py

What I’ve done
Set up the Hugging Face environment (transformers + datasets)
Loaded pretrained BERT (bert-base-uncased)
Loaded a subset of Wikipedia dataset
Cleaned and filtered the dataset (removed empty text)
Tokenized the text using BERT tokenizer
Grouped tokens into fixed-length sequences
Implemented baseline masking (15% MLM) using Hugging Face collator
Set up the training pipeline using Trainer API
Successfully ran baseline BERT training + evaluation

What’s left now is:

implement the custom SpanMaskingDataCollator
run training with span masking
compare baseline vs span-masked BERT
do the experiments on span length and masking rate if time allows
