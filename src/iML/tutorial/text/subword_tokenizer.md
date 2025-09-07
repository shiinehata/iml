Subword tokenizers
This tutorial demonstrates how to generate a subword vocabulary from a dataset, and use it to build a text.BertTokenizer from the vocabulary.

The main advantage of a subword tokenizer is that it interpolates between word-based and character-based tokenization. Common words get a slot in the vocabulary, but the tokenizer can fall back to word pieces and individual characters for unknown words.
Overview

The tensorflow_text package includes TensorFlow implementations of many common tokenizers. This includes three subword-style tokenizers:

    text.BertTokenizer - The BertTokenizer class is a higher level interface. It includes BERT's token splitting algorithm and a WordPieceTokenizer. It takes sentences as input and returns token-IDs.
    text.WordpieceTokenizer - The WordPieceTokenizer class is a lower level interface. It only implements the WordPiece algorithm. You must standardize and split the text into words before calling it. It takes words as input and returns token-IDs.
    text.SentencepieceTokenizer - The SentencepieceTokenizer requires a more complex setup. Its initializer requires a pre-trained sentencepiece model. See the google/sentencepiece repository for instructions on how to build one of these models. It can accept sentences as input when tokenizing.

This tutorial builds a Wordpiece vocabulary in a top down manner, starting from existing words. This process doesn't work for Japanese, Chinese, or Korean since these languages don't have clear multi-character units. To tokenize these languages consider using text.SentencepieceTokenizer, text.UnicodeCharTokenizer or this approach.
Setup

pip install -q -U "tensorflow-text==2.11.\*"

pip install -q tensorflow_datasets

import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

Download the dataset

Fetch the Portuguese/English translation dataset from tfds:

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

This dataset produces Portuguese/English sentence pairs:

for pt, en in train_examples.take(1):
print("Portuguese: ", pt.numpy().decode('utf-8'))
print("English: ", en.numpy().decode('utf-8'))

Note a few things about the example sentences above:

    They're lower case.
    There are spaces around the punctuation.
    It's not clear if or what unicode normalization is being used.

train_en = train_examples.map(lambda pt, en: en)
train_pt = train_examples.map(lambda pt, en: pt)

Generate the vocabulary
This section generates a wordpiece vocabulary from a dataset. If you already have a vocabulary file and just want to see how to build a text.BertTokenizer or text.WordpieceTokenizer tokenizer with it then you can skip ahead to the Build the tokenizer section.
The vocabulary generation code is included in the tensorflow_text pip package. It is not imported by default , you need to manually import it:

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

The bert_vocab.bert_vocab_from_dataset function will generate the vocabulary.

There are many arguments you can set to adjust its behavior. For this tutorial, you'll mostly use the defaults. If you want to learn more about the options, first read about the algorithm, and then have a look at the code.

This takes about 2 minutes.
bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict( # The target vocabulary size
vocab_size = 8000, # Reserved tokens that must be included in the vocabulary
reserved_tokens=reserved_tokens, # Arguments for `text.BertTokenizer`
bert_tokenizer_params=bert_tokenizer_params, # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
learn_params={},
)

%%time
pt_vocab = bert_vocab.bert_vocab_from_dataset(
train_pt.batch(1000).prefetch(2),
\*\*bert_vocab_args
)
Here are some slices of the resulting vocabulary.

print(pt_vocab[:10])
print(pt_vocab[100:110])
print(pt_vocab[1000:1010])
print(pt_vocab[-10:])
Write a vocabulary file:

def write_vocab_file(filepath, vocab):
with open(filepath, 'w') as f:
for token in vocab:
print(token, file=f)

write_vocab_file('pt_vocab.txt', pt_vocab)
Use that function to generate a vocabulary from the english data:

%%time
en_vocab = bert_vocab.bert_vocab_from_dataset(
train_en.batch(1000).prefetch(2),
\*\*bert_vocab_args
)
print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])
Here are the two vocabulary files:

write_vocab_file('en_vocab.txt', en_vocab)
Build the tokenizer

The text.BertTokenizer can be initialized by passing the vocabulary file's path as the first argument (see the section on tf.lookup for other options):

pt_tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)

Now you can use it to encode some text. Take a batch of 3 examples from the english data:

for pt_examples, en_examples in train_examples.batch(3).take(1):
for ex in en_examples:
print(ex.numpy())
Run it through the BertTokenizer.tokenize method. Initially, this returns a tf.RaggedTensor with axes (batch, word, word-piece):

# Tokenize the examples -> (batch, word, word-piece)

token_batch = en_tokenizer.tokenize(en_examples)

# Merge the word and word-piece axes -> (batch, tokens)

token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
print(ex)
If you replace the token IDs with their text representations (using tf.gather) you can see that in the first example the words "searchability" and "serendipity" have been decomposed into "search ##ability" and "s ##ere ##nd ##ip ##ity":

# Lookup each token id in the vocabulary.

txt_tokens = tf.gather(en_vocab, token_batch)

# Join with spaces.

tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)
To re-assemble words from the extracted tokens, use the BertTokenizer.detokenize method:

words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)
