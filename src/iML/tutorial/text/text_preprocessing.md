BERT Preprocessing with TF Text
Overview

Text preprocessing is the end-to-end transformation of raw text into a model’s integer inputs. NLP models are often accompanied by several hundreds (if not thousands) of lines of Python code for preprocessing text. Text preprocessing is often a challenge for models because:

    Training-serving skew. It becomes increasingly difficult to ensure that the preprocessing logic of the model's inputs are consistent at all stages of model development (e.g. pretraining, fine-tuning, evaluation, inference). Using different hyperparameters, tokenization, string preprocessing algorithms or simply packaging model inputs inconsistently at different stages could yield hard-to-debug and disastrous effects to the model.

    Efficiency and flexibility. While preprocessing can be done offline (e.g. by writing out processed outputs to files on disk and then reconsuming said preprocessed data in the input pipeline), this method incurs an additional file read and write cost. Preprocessing offline is also inconvenient if there are preprocessing decisions that need to happen dynamically. Experimenting with a different option would require regenerating the dataset again.

    Complex model interface. Text models are much more understandable when their inputs are pure text. It's hard to understand a model when its inputs require an extra, indirect encoding step. Reducing the preprocessing complexity is especially appreciated for model debugging, serving, and evaluation.

Additionally, simpler model interfaces also make it more convenient to try the model (e.g. inference or training) on different, unexplored datasets.
Text preprocessing with TF.Text

Using TF.Text's text preprocessing APIs, we can construct a preprocessing function that can transform a user's text dataset into the model's integer inputs. Users can package preprocessing directly as part of their model to alleviate the above mentioned problems.

This tutorial will show how to use TF.Text preprocessing ops to transform text data into inputs for the BERT model and inputs for language masking pretraining task described in "Masked LM and Masking Procedure" of BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. The process involves tokenizing text into subword units, combining sentences, trimming content to a fixed size and extracting labels for the masked language modeling task.

Setup

Let's import the packages and libraries we need first.

pip install -q -U "tensorflow-text==2.11.\*"

import tensorflow as tf
import tensorflow_text as text
import functools
print("TensorFlow version: ", tf.**version**)

Our data contains two text features and we can create a example tf.data.Dataset. Our goal is to create a function that we can supply Dataset.map() with to be used in training.

examples = {
"text_a": [
"Sponge bob Squarepants is an Avenger",
"Marvel Avengers"
],
"text_b": [
"Barack Obama is the President.",
"President is the highest office"
],
}

dataset = tf.data.Dataset.from_tensor_slices(examples)
next(iter(dataset))

Tokenizing

Our first step is to run any string preprocessing and tokenize our dataset. This can be done using the text.BertTokenizer, which is a text.Splitter that can tokenize sentences into subwords or wordpieces for the BERT model given a vocabulary generated from the Wordpiece algorithm. You can learn more about other subword tokenizers available in TF.Text from here.

The vocabulary can be from a previously generated BERT checkpoint, or you can generate one yourself on your own data. For the purposes of this example, let's create a toy vocabulary:

\_VOCAB = [ # Special tokens
b"[UNK]", b"[MASK]", b"[RANDOM]", b"[CLS]", b"[SEP]", # Suffixes
b"##ack", b"##ama", b"##ger", b"##gers", b"##onge", b"##pants", b"##uare",
b"##vel", b"##ven", b"an", b"A", b"Bar", b"Hates", b"Mar", b"Ob",
b"Patrick", b"President", b"Sp", b"Sq", b"bob", b"box", b"has", b"highest",
b"is", b"office", b"the",
]

\_START_TOKEN = \_VOCAB.index(b"[CLS]")
\_END_TOKEN = \_VOCAB.index(b"[SEP]")
\_MASK_TOKEN = \_VOCAB.index(b"[MASK]")
\_RANDOM_TOKEN = \_VOCAB.index(b"[RANDOM]")
\_UNK_TOKEN = \_VOCAB.index(b"[UNK]")
\_MAX_SEQ_LEN = 8
\_MAX_PREDICTIONS_PER_BATCH = 5

\_VOCAB_SIZE = len(\_VOCAB)

lookup_table = tf.lookup.StaticVocabularyTable(
tf.lookup.KeyValueTensorInitializer(
keys=\_VOCAB,
key_dtype=tf.string,
values=tf.range(
tf.size(\_VOCAB, out_type=tf.int64), dtype=tf.int64),
value_dtype=tf.int64
),
num_oov_buckets=1
)

Let's construct a text.BertTokenizer using the above vocabulary and tokenize the text inputs into a RaggedTensor.`.

bert_tokenizer = text.BertTokenizer(lookup_table, token_out_type=tf.string)
bert_tokenizer.tokenize(examples["text_a"])
bert_tokenizer.tokenize(examples["text_b"])

Text output from text.BertTokenizer allows us see how the text is being tokenized, but the model requires integer IDs. We can set the token_out_type param to tf.int64 to obtain integer IDs (which are the indices into the vocabulary).

bert_tokenizer = text.BertTokenizer(lookup_table, token_out_type=tf.int64)
segment_a = bert_tokenizer.tokenize(examples["text_a"])
segment_a

segment_b = bert_tokenizer.tokenize(examples["text_b"])
segment_b

text.BertTokenizer returns a RaggedTensor with shape [batch, num_tokens, num_wordpieces]. Because we don't need the extra num_tokens dimensions for our current use case, we can merge the last two dimensions to obtain a RaggedTensor with shape [batch, num_wordpieces]:

segment_a = segment_a.merge_dims(-2, -1)
segment_a

segment_b = segment_b.merge_dims(-2, -1)
segment_b

Content Trimming

The main input to BERT is a concatenation of two sentences. However, BERT requires inputs to be in a fixed-size and shape and we may have content which exceed our budget.

We can tackle this by using a text.Trimmer to trim our content down to a predetermined size (once concatenated along the last axis). There are different text.Trimmer types which select content to preserve using different algorithms. text.RoundRobinTrimmer for example will allocate quota equally for each segment but may trim the ends of sentences. text.WaterfallTrimmer will trim starting from the end of the last sentence.

For our example, we will use RoundRobinTrimmer which selects items from each segment in a left-to-right manner.

trimmer = text.RoundRobinTrimmer(max_seq_length=\_MAX_SEQ_LEN)
trimmed = trimmer.trim([segment_a, segment_b])
trimmed

trimmed now contains the segments where the number of elements across a batch is 8 elements (when concatenated along axis=-1).
Combining segments

Now that we have segments trimmed, we can combine them together to get a single RaggedTensor. BERT uses special tokens to indicate the beginning ([CLS]) and end of a segment ([SEP]). We also need a RaggedTensor indicating which items in the combined Tensor belong to which segment. We can use text.combine_segments() to get both of these Tensor with special tokens inserted.

segments_combined, segments_ids = text.combine_segments(
trimmed,
start_of_sequence_id=\_START_TOKEN, end_of_segment_id=\_END_TOKEN)
segments_combined, segments_ids

Masked Language Model Task

Now that we have our basic inputs, we can begin to extract the inputs needed for the "Masked LM and Masking Procedure" task described in BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

The masked language model task has two sub-problems for us to think about: (1) what items to select for masking and (2) what values are they assigned?
Item Selection

Because we will choose to select items randomly for masking, we will use a text.RandomItemSelector. RandomItemSelector randomly selects items in a batch subject to restrictions given (max_selections_per_batch, selection_rate and unselectable_ids) and returns a boolean mask indicating which items were selected.

random_selector = text.RandomItemSelector(
max_selections_per_batch=\_MAX_PREDICTIONS_PER_BATCH,
selection_rate=0.2,
unselectable_ids=[_START_TOKEN, _END_TOKEN, _UNK_TOKEN]
)
selected = random_selector.get_selection_mask(
segments_combined, axis=1)
selected

Choosing the Masked Value

The methodology described the original BERT paper for choosing the value for masking is as follows:

For mask_token_rate of the time, replace the item with the [MASK] token:

"my dog is hairy" -> "my dog is [MASK]"

For random_token_rate of the time, replace the item with a random word:

"my dog is hairy" -> "my dog is apple"

For 1 - mask_token_rate - random_token_rate of the time, keep the item unchanged:

"my dog is hairy" -> "my dog is hairy."

text.MaskedValuesChooser encapsulates this logic and can be used for our preprocessing function. Here's an example of what MaskValuesChooser returns given a mask_token_rate of 80% and default random_token_rate:

mask_values_chooser = text.MaskValuesChooser(\_VOCAB_SIZE, \_MASK_TOKEN, 0.8)
mask_values_chooser.get_mask_values(segments_combined)

When supplied with a RaggedTensor input, text.MaskValuesChooser returns a RaggedTensor of the same shape with either \_MASK_VALUE (0), a random ID, or the same unchanged id.
Generating Inputs for Masked Language Model Task

Now that we have a RandomItemSelector to help us select items for masking and text.MaskValuesChooser to assign the values, we can use text.mask_language_model() to assemble all the inputs of this task for our BERT model.

masked_token_ids, masked_pos, masked_lm_ids = text.mask_language_model(
segments_combined,
item_selector=random_selector, mask_values_chooser=mask_values_chooser)

Let's dive deeper and examine the outputs of mask_language_model(). The output of masked_token_ids is:

masked_token_ids

Remember that our input is encoded using a vocabulary. If we decode masked_token_ids using our vocabulary, we get:

tf.gather(\_VOCAB, masked_token_ids)

Notice that some wordpiece tokens have been replaced with either [MASK], [RANDOM] or a different ID value. masked_pos output gives us the indices (in the respective batch) of the tokens that have been replaced.

masked_pos
masked_lm_ids gives us the original value of the token.

masked_lm_ids
We can again decode the IDs here to get human readable values.

tf.gather(\_VOCAB, masked_lm_ids)
Padding Model Inputs

Now that we have all the inputs for our model, the last step in our preprocessing is to package them into fixed 2-dimensional Tensors with padding and also generate a mask Tensor indicating the values which are pad values. We can use text.pad_model_inputs() to help us with this task.

# Prepare and pad combined segment inputs

input*word_ids, input_mask = text.pad_model_inputs(
masked_token_ids, max_seq_length=\_MAX_SEQ_LEN)
input_type_ids, * = text.pad_model_inputs(
segments_ids, max_seq_length=\_MAX_SEQ_LEN)

# Prepare and pad masking task inputs

masked*lm_positions, masked_lm_weights = text.pad_model_inputs(
masked_pos, max_seq_length=\_MAX_PREDICTIONS_PER_BATCH)
masked_lm_ids, * = text.pad_model_inputs(
masked_lm_ids, max_seq_length=\_MAX_PREDICTIONS_PER_BATCH)

model_inputs = {
"input_word_ids": input_word_ids,
"input_mask": input_mask,
"input_type_ids": input_type_ids,
"masked_lm_ids": masked_lm_ids,
"masked_lm_positions": masked_lm_positions,
"masked_lm_weights": masked_lm_weights,
}
model_inputs
