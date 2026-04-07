 steps to run :  
 cd ~/Desktop/bert_project
pip3 install transformers datasets accelerate
python3 baseline_mlm.py

What’s left now is:

implement the custom SpanMaskingDataCollator
run training with span masking
compare baseline vs span-masked BERT
do the experiments on span length and masking rate if time allows
