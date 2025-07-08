# Natural Language Processing (NLP)

## What is "self-supervised learning"?
Self-supervised learning: Training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

## What is a "language model"?
Model trained to guess the next word in a text (having read the ones before)

## Why is a language model considered self-supervised?
Due to the existance of the labeling in the training data itself. For example, for a language model the labels are the words in the data.

## What are self-supervised models usually used for?
Normally they are used to develop pretrained models for transfer learning.

## Why do we fine-tune language models?
Because training a language model from scratch can be too expensive and it's easier to apply transfer learning for our specific task.

## What are the three steps to create a state-of-the-art text classifier?
1. Choose pretrained model
2. Fine tune
3. Save
4. Check solution cause this is wrong XD
> 1. Train a language model on a large corpus of text (already done for ULM-FiT by Sebastian Ruder and Jeremy!)
> 2. Fine-tune the language model on text classification dataset
> 3. Fine-tune the language model as a text classifier instead.

## How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
With fine tuning the sequence to sequence model first from Wikipedia English to IMDb English. It helps to the model to get used to the style of the corpus we are targetting.

## What are the three steps to prepare your data for a language model?
NPI
> 1. Tokenization
> 2. Numericalization
> 3. Language model DataLoader

## What is "tokenization"? Why do we need it?
The first step to convert text to input numbers for a neural network. It consists in splitting said text in words or subwords (it depends on the tokenizer).

## Name three different approaches to tokenization.
Word, character and subword based tokenizations.

## What is xxbos?
In the example in the book it was a special token that indicated the beginning of a new text.
> This is a special token added by fastai that indicated the beginning of the text.

## List four rules that fastai applies to text during tokenization.
> Here are all the rules:  
> fix_html :: replace special HTML characters by a readable version (IMDb reviews have quite a few of them for instance) ;  
> replace_rep :: replace any character repeated three times or more by a special token for repetition (xxrep), the number of times it’s repeated, then the character ;  
> replace_wrep :: replace any word repeated three times or more by a special token for word repetition (xxwrep), the number of times it’s repeated, then the word ;  
> spec_add_spaces :: add spaces around / and # ;  
> rm_useless_spaces :: remove all repetitions of the space character ;  
> replace_all_caps :: lowercase a word written in all caps and adds a special token for all caps (xxcap) in front of it ;  
> replace_maj :: lowercase a capitalized word and adds a special token for capitalized (xxmaj) in front of it ;  
> lowercase :: lowercase all text and adds a special token at the beginning (xxbos) and/or the end (xxeos).

## Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
In order to encode more efficiently?
> We can expect that repeated characters could have special or different meaning than just a single character. By replacing them with a special token showing the number of repetitions, the model’s embedding matrix can encode information about general concepts such as repeated characters rather than requiring a separate token for every number of repetitions of every character.

## What is "numericalization"?
Conversion of tokens to input IDs.

## Why might there be words that are replaced with the "unknown word" token?
Because there are too many words that exist in any language that we would have to encode to build a complete language vocabulary. This big vocabularies are the reason why word level encoding is not used.

## With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
> a. The dataset is split into 64 mini-streams (batch size)  
> b. Each batch has 64 rows (batch size) and 64 columns (sequence length)  
> c. The first row of the first batch contains the beginning of the first mini-stream (tokens 1-64)  
> d. The second row of the first batch contains the beginning of the second mini-stream  
> e. The first row of the second batch contains the second chunk of the first mini-stream (tokens 65-128)  

## Why do we need padding for text classification? Why don't we need it for language modeling?
Because we have models that require input batches of the same size. Nfc

## What does an embedding matrix for NLP contain? What is its shape?
> It contains vector representations of all tokens in the vocabulary. The embedding matrix has the size (vocab_size x embedding_size), where vocab_size is the length of the vocabulary, and embedding_size is an arbitrary number defining the number of latent factors of the tokens.

## What is "perplexity"?
> Perplexity is a commonly used metric in NLP for language models. It is the exponential of the loss.

## Why do we have to pass the vocabulary of the language model to the classifier data block?
> This is to ensure the same correspondence of tokens to index so the model can appropriately use the embeddings learned during LM fine-tuning.

## What is "gradual unfreezing"?
> This refers to unfreezing one layer at a time and fine-tuning the pretrained model.

## Why is text generation always likely to be ahead of automatic identification of machine-generated texts?
> The classification models could be used to improve text generation algorithms (evading the classifier) so the text generation algorithms will always be ahead.
