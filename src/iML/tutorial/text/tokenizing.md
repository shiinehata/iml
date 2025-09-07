Tokenizing with TF Text
Overview

Tokenization is the process of breaking up a string into tokens. Commonly, these tokens are words, numbers, and/or punctuation. The tensorflow_text package provides a number of tokenizers available for preprocessing text required by your text-based models. By performing the tokenization in the TensorFlow graph, you will not need to worry about differences between the training and inference workflows and managing preprocessing scripts.

This guide discusses the many tokenization options provided by TensorFlow Text, when you might want to use one option over another, and how these tokenizers are called from within your model.
Setup

pip install -q "tensorflow-text==2.11.\*"

import requests
import tensorflow as tf
import tensorflow_text as tf_text
Splitter API

The main interfaces are Splitter and SplitterWithOffsets which have single methods split and split_with_offsets. The SplitterWithOffsets variant (which extends Splitter) includes an option for getting byte offsets. This allows the caller to know which bytes in the original string the created token was created from.

The Tokenizer and TokenizerWithOffsets are specialized versions of the Splitter that provide the convenience methods tokenize and tokenize_with_offsets respectively.

Generally, for any N-dimensional input, the returned tokens are in a N+1-dimensional RaggedTensor with the inner-most dimension of tokens mapping to the original individual strings.

class Splitter {
@abstractmethod
def split(self, input)
}

class SplitterWithOffsets(Splitter) {
@abstractmethod
def split_with_offsets(self, input)
}

There is also a Detokenizer interface. Any tokenizer implementing this interface can accept a N-dimensional ragged tensor of tokens, and normally returns a N-1-dimensional tensor or ragged tensor that has the given tokens assembled together.

class Detokenizer {
@abstractmethod
def detokenize(self, input)
}

Tokenizers

Below is the suite of tokenizers provided by TensorFlow Text. String inputs are assumed to be UTF-8. Please review the Unicode guide for converting strings to UTF-8.
Whole word tokenizers

These tokenizers attempt to split a string by words, and is the most intuitive way to split text.
WhitespaceTokenizer

The text.WhitespaceTokenizer is the most basic tokenizer which splits strings on ICU defined whitespace characters (eg. space, tab, new line). This is often good for quickly building out prototype models.

tokenizer = tf_text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())

You may notice a shortcome of this tokenizer is that punctuation is included with the word to make up a token. To split the words and punctuation into separate tokens, the UnicodeScriptTokenizer should be used.
UnicodeScriptTokenizer

The UnicodeScriptTokenizer splits strings based on Unicode script boundaries. The script codes used correspond to International Components for Unicode (ICU) UScriptCode values. See: http://icu-project.org/apiref/icu4c/uscript_8h.html

In practice, this is similar to the WhitespaceTokenizer with the most apparent difference being that it will split punctuation (USCRIPT_COMMON) from language texts (eg. USCRIPT_LATIN, USCRIPT_CYRILLIC, etc) while also separating language texts from each other. Note that this will also split contraction words into separate tokens.

tokenizer = tf_text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())

Subword tokenizers

Subword tokenizers can be used with a smaller vocabulary, and allow the model to have some information about novel words from the subwords that make create it.

We briefly discuss the Subword tokenization options below, but the Subword Tokenization tutorial goes more in depth and also explains how to generate the vocab files.
WordpieceTokenizer

WordPiece tokenization is a data-driven tokenization scheme which generates a set of sub-tokens. These sub tokens may correspond to linguistic morphemes, but this is often not the case.

The WordpieceTokenizer expects the input to already be split into tokens. Because of this prerequisite, you will often want to split using the WhitespaceTokenizer or UnicodeScriptTokenizer beforehand.

tokenizer = tf_text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())
After the string is split into tokens, the WordpieceTokenizer can be used to split into subtokens.

url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
r = requests.get(url)
filepath = "vocab.txt"
open(filepath, 'wb').write(r.content)

subtokenizer = tf_text.UnicodeScriptTokenizer(filepath)
subtokens = tokenizer.tokenize(tokens)
print(subtokens.to_list())

BertTokenizer

The BertTokenizer mirrors the original implementation of tokenization from the BERT paper. This is backed by the WordpieceTokenizer, but also performs additional tasks such as normalization and tokenizing to words first.

tokenizer = tf_text.BertTokenizer(filepath, token_out_type=tf.string, lower_case=True)
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())
SentencepieceTokenizer

The SentencepieceTokenizer is a sub-token tokenizer that is highly configurable. This is backed by the Sentencepiece library. Like the BertTokenizer, it can include normalization and token splitting before splitting into sub-tokens.

url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_oss_model.model?raw=true"
sp_model = requests.get(url).content

tokenizer = tf_text.SentencepieceTokenizer(sp_model, out_type=tf.string)
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())
Other splitters
UnicodeCharTokenizer

This splits a string into UTF-8 characters. It is useful for CJK languages that do not have spaces between words.

tokenizer = tf_text.UnicodeCharTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())
The output is Unicode codepoints. This can be also useful for creating character ngrams, such as bigrams. To convert back into UTF-8 characters.

characters = tf.strings.unicode_encode(tf.expand_dims(tokens, -1), "UTF-8")
bigrams = tf_text.ngrams(characters, 2, reduction_type=tf_text.Reduction.STRING_JOIN, string_separator='')
print(bigrams.to_list())
HubModuleTokenizer

This is a wrapper around models deployed to TF Hub to make the calls easier since TF Hub currently does not support ragged tensors. Having a model perform tokenization is particularly useful for CJK languages when you want to split into words, but do not have spaces to provide a heuristic guide. At this time, we have a single segmentation model for Chinese.

MODEL_HANDLE = "https://tfhub.dev/google/zh_segmentation/1"
segmenter = tf_text.HubModuleTokenizer(MODEL_HANDLE)
tokens = segmenter.tokenize(["新华社北京"])
print(tokens.to_list())
It may be difficult to view the results of the UTF-8 encoded byte strings. Decode the list values to make viewing easier.

def decode_list(x):
if type(x) is list:
return list(map(decode_list, x))
return x.decode("UTF-8")

def decode_utf8_tensor(x):
return list(map(decode_list, x.to_list()))

print(decode_utf8_tensor(tokens))
SplitMergeTokenizer

The SplitMergeTokenizer & SplitMergeFromLogitsTokenizer have a targeted purpose of splitting a string based on provided values that indicate where the string should be split. This is useful when building your own segmentation models like the previous Segmentation example.

For the SplitMergeTokenizer, a value of 0 is used to indicate the start of a new string, and the value of 1 indicates the character is part of the current string.

strings = ["新华社北京"]
labels = [[0, 1, 1, 0, 1]]
tokenizer = tf_text.SplitMergeTokenizer()
tokens = tokenizer.tokenize(strings, labels)
print(decode_utf8_tensor(tokens))
The SplitMergeFromLogitsTokenizer is similar, but it instead accepts logit value pairs from a neural network that predict if each character should be split into a new string or merged into the current one.

strings = [["新华社北京"]]
labels = [[[5.0, -3.2], [0.2, 12.0], [0.0, 11.0], [2.2, -1.0], [-3.0, 3.0]]]
tokenizer = tf_text.SplitMergeFromLogitsTokenizer()
tokenizer.tokenize(strings, labels)
print(decode_utf8_tensor(tokens))
RegexSplitter

The RegexSplitter is able to segment strings at arbitrary breakpoints defined by a provided regular expression.

splitter = tf_text.RegexSplitter("\s")
tokens = splitter.split(["What you know you can't explain, but you feel it."], )
print(tokens.to_list())
Offsets

When tokenizing strings, it is often desired to know where in the original string the token originated from. For this reason, each tokenizer which implements TokenizerWithOffsets has a tokenize_with_offsets method that will return the byte offsets along with the tokens. The start_offsets lists the bytes in the original string each token starts at, and the end_offsets lists the bytes immediately after the point where each token ends. To refrase, the start offsets are inclusive and the end offsets are exclusive.

tokenizer = tf_text.UnicodeScriptTokenizer()
(tokens, start_offsets, end_offsets) = tokenizer.tokenize_with_offsets(['Everything not saved will be lost.'])
print(tokens.to_list())
print(start_offsets.to_list())
print(end_offsets.to_list())
Detokenization

Tokenizers which implement the Detokenizer provide a detokenize method which attempts to combine the strings. This has the chance of being lossy, so the detokenized string may not always match exactly the original, pre-tokenized string.

tokenizer = tf_text.UnicodeCharTokenizer()
tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
print(tokens.to_list())
strings = tokenizer.detokenize(tokens)
print(strings.numpy())
TF Data

TF Data is a powerful API for creating an input pipeline for training models. Tokenizers work as expected with the API.

docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
tokenizer = tf_text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = iter(tokenized_docs)
print(next(iterator).to_list())
print(next(iterator).to_list())
