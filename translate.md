# my running params:
(pls try this one too -replace_unk)
 -batch_size 20 -beam_size 5 -model models/news_step_20.pt -src data/news/test.article.txt -output output/output.txt -min_length 5 -verbose -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 -verbose -block_ngram_repeat 3 -ignore_when_blocking "." "</t>" "<t>" 

translate.py
# Options: translate.py:
translate.py

### **Model**:
* **-model []** 
Path to model .pt file

### **Data**:
* **-data_type [text]** 
Type of the source input. Options: [text|img].

* **-src []** 
Source sequence to decode (one line per sequence)

* **-src_dir []** 
Source directory for image or audio files

* **-tgt []** 
True target sequence (optional)

* **-output [pred.txt]** 
Path to output the predictions (each line will be the decoded sequence

* **-report_bleu []** 
Report bleu score after translation, call tools/multi-bleu.perl on command line

* **-report_rouge []** 
Report rouge 1/2/3/L/SU4 score after translation call tools/test_rouge.py on
command line

* **-dynamic_dict []** 
Create dynamic dictionaries

* **-share_vocab []** 
Share source and target vocabulary

### **Beam**:
* **-fast []** 
Use fast beam search (some features may not be supported!)

* **-beam_size [5]** 
Beam size

* **-min_length []** 
Minimum prediction length

* **-max_length [100]** 
Maximum prediction length.

* **-max_sent_length []** 
Deprecated, use `-max_length` instead

* **-stepwise_penalty []** 
Apply penalty at every decoding step. Helpful for summary penalty.

* **-length_penalty [none]** 
Length Penalty to use.

* **-coverage_penalty [none]** 
Coverage Penalty to use.

* **-alpha []** 
Google NMT length penalty parameter (higher = longer generation)

* **-beta []** 
Coverage penalty parameter

* **-block_ngram_repeat []** 
Block repetition of ngrams during decoding.

* **-ignore_when_blocking []** 
Ignore these strings when blocking repeats. You want to block sentence
delimiters.

* **-replace_unk []** 
Replace the generated UNK tokens with the source token that had highest
attention weight. If phrase_table is provided, it will lookup the identified
source token and give the corresponding target token. If it is not provided(or
the identified source token does not exist in the table) then it will copy the
source token

### **Logging**:
* **-verbose []** 
Print scores and predictions for each sentence

* **-log_file []** 
Output logs to a file under this path.

* **-attn_debug []** 
Print best attn for each word

* **-dump_beam []** 
File to dump beam information to.

* **-n_best [1]** 
If verbose is set, will output the n_best decoded sentences

### **Efficiency**:
* **-batch_size [30]** 
Batch size

* **-gpu [-1]** 
Device to run on

### **Speech**:
* **-sample_rate [16000]** 
Sample rate.

* **-window_size [0.02]** 
Window size for spectrogram in seconds

* **-window_stride [0.01]** 
Window stride for spectrogram in seconds

* **-window [hamming]** 
Window type for spectrogram generation
