
# Neural Machine Translation

A simple seq2seq model, with and without attention, is implemented. The model is trained with teacher-forcing and decoding is done using beam search to get the best possible translation. The source language is French and the target language is English.

# DataSet

The main corpus for this project comes from the oﬃcial records (Hansards) of the 36th Canadian Parliament, including debates from both the House of Representatives and the Senate. This corpus is available at /data/Hansard/ and has been split into Training/ and Testing/ directories. This data set consists of pairs of corresponding ﬁles (*.e is the English equivalent of the French *.f) in which every line is a sentence. Here, sentence are aligned in both the files. That is, the nth sentence in one ﬁle corresponds to the nth sentence in its corresponding ﬁle (e.g., line n in fubar.e is aligned with line n in fubar.f).

# Dependencies

### Install
The Code is written in Python 3.7 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.The code is written in PyTorch version 1.2.0 (https://pytorch.org/docs/1.2.0/), 

To install pip run in the command Line
```
python -m ensurepip -- default-pip
``` 
to upgrade it 
```
python -m pip install -- upgrade pip setuptools wheel
```
to upgrade Python
```
pip install python -- upgrade
```

# Usage
You can clone the repository using 
```
https://github.com/shasha3493/Neural-Machine-Translation-French-to-English-.git
```
Then run the following code block line-by-line from your working directory. In order, it:

1. Builds maps between words and unique numerical identiﬁers for each language.
2. Splits the training data into a portion to train on and a hold-out portion.
3. Trains the encoder/decoder without attention and stores the model parameters.
4. Trains the encoder/decoder with attention and stores the model parameters.
5. Returns the average BLEU score of the encoder/decoder without attention on the test set.
6. Returns the average BLEU score of the encoder/decoder with attention on the test set.

TRAIN=.../Training/ (Path to the Training dataset)

TEST=.../Testing/  (Path to the Test dataset

1)

python3.7 a2_run.py vocab $TRAIN e vocab.e.gz

python3.7 a2_run.py vocab $TRAIN f vocab.f.gz  

2)

python3.7 a2_run.py split $TRAIN train.txt.gz dev.txt.gz 

3)

python3.7 a2_run.py train $TRAIN \ vocab.e.gz vocab.f.gz \ train.txt.gz dev.txt.gz \ model_wo_att.pt.gz \ --device cuda 

4)

python3.7 a2_run.py train $TRAIN \ vocab.e.gz vocab.f.gz \ train.txt.gz dev.txt.gz \ model_w_att.pt.gz \ --with-attention \ --device cuda 

5)

python3.7 a2_run.py test $TEST \ vocab.e.gz vocab.f.gz model_wo_att.pt.gz \ --device cuda 

6)

python3.7 a2_run.py test $TEST \ vocab.e.gz vocab.f.gz model_w_att.pt.gz \ --with-attention --device cuda



# Results

Number of Training Epochs = 5

### Without Attention:

#### Training
Loss = 13.43
Bleu Score = 29%

#### Testing
Bleu Score = 33%

### With Attention:

#### Training
Loss = 10.04
Bleu Score = 33%

#### Testing
Bleu Score = 38%






```python

```
