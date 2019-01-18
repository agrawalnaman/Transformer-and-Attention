# Transformer-and-Attention
##### **By [Naman Agrawal](https://github.com/agrawalnaman) & Priyanka Cornelius**

This blog is aimed at explaining the [Transformer and it's Attention mechanism](https://arxiv.org/abs/1706.03762) in a lucid and intuitive manner.
## Prerequisites:
To get most out of this post it is recommended that you are comfortable and acquainted with these terms :

+ **RNN - [Andrej Karpathy’s blog The Unreasonable Effectiveness of Recurrent Neural Networks ](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)**
+ **Seq2Seq - [Nathan Lintz Sequence Modeling With Neural Networks (Part 1): Language & Seq2Seq ](https://indico.io/blog/sequence-modeling-neuralnets-part1/)**
+ **LSTM - [Christopher Olah’s blog Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)** 
+ **Attention - [Christopher Olah Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/#attentional-interfaces)** 

## Let's start with the basics:
### Why do we need a Transformer?
###### reffered from [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations. End- to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple- language question answering and language modeling tasks.

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.

### Why the name Transformer?
The Transformer architecture is aimed at the problem of [sequence transduction (by Alex Graves)](https://arxiv.org/abs/1211.3711), **meaning any task where input sequences are transformed into output sequences**. This includes speech recognition, text-to-speech transformation, machine translation, protein secondary structure prediction, Turing machines etc. Basically the goal is to design a single framework to handle as many sequences as possible.
### What does a Transformer do?
+ Transformer is based on sequence-to-sequence model for Statistical Machine Translation (SMT) as introduced in [Cho et al., 2014](https://arxiv.org/abs/1406.1078) . It includes two RNNs, one for encoder to process the input and the other as a decoder, for generating the output.

+ In general, transformer’s encoder maps input sequence to its continuous representation z which in turn is used by decoder to generate output, one symbol at a time.

+ The final state of the encoder is a fixed size vector z that must encode entire source sentence which includes the sentence meaning. This final state is therefore called sentence embedding1.

+ The encoder-decoder model is designed at its each step to be auto-regressive - i.e. use previously generated symbols as extra input while generating next symbol. Thus, xi+yi−1→yi

### To get a deeper insight into the Transformer in a more illustrated format we read [The illustrated Transformer - by Jay Alammar](https://jalammar.github.io/illustrated-transformer/), However we were left curious with a few unanswered questions after reading it. 
### In this Blog we will attempt to answer those questions.

### What is Attention?
###### [AttentionPrimer](https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#attention-basis)

Attention between encoder and decoder is crucial in NMT.Attention is a function that maps the 2-element input (query, key-value pairs) to an output. The output given by the mapping function is a weighted sum of the values. Where weights for each value measures how much each input key interacts with (or answers) the query. While the attention is a goal for many research, the novelty about transformer attention is that it is multi-head self-attention.
#### Basic Idea: (Bahdanau et al. 2015)
+ Encode each word in the sentence into a vector 
+ When decoding, perform a linear combination of these vectors, weighted by “attention weights” 
+ Use this combination in picking the next word

## After reading the above expalination our major concern was, what is a KEY, QUERY and VALUE?

In terms of encoder-decoder, the query is usually the hidden state of the decoder. Whereas key, is the hidden state of the encoder, and the corresponding value is normalized weight, representing how much attention a key gets. Output is calculated as a wighted sum – here the dot product of query and key is used to get a value.

It is assumed that queries and keys are of dk dimension and values are of dv dimension. Those dimensions are imposed by the linear projection discussed in the multi-head attention section. The input is represented by three matrices: queries’ matrix Q, keys’ matrix K and values’ matrix V.

The compatibility function (see Attention primer) is considered in terms of two, additive and multiplicative (dot-product) variants Bahdanau et al. 2014  with similar theoretical complexity.

#### Reffered from [CMU CS 11-747, Spring 2018 Graham Neubig](http://www.phontron.com/class/nn4nlp2018/schedule/attention.html)

![encoder-decoder](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/Encoder-decoder-models.png)

![calculating-attention-1](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/calculating-attention-.png)

![calculating-attention-2](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/calculating-attention-2.png)


