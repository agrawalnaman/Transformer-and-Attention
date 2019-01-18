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
+ Transformer is based on sequence-to-sequence model for Statistical Machine Translation (SMT) as introduced in Cho et al., 2014 . It includes two RNNs, one for encoder to process the input and the other as a decoder, for generating the output.

+ In general, transformer’s encoder maps input sequence to its continuous representation z which in turn is used by decoder to generate output, one symbol at a time.

+ The final state of the encoder is a fixed size vector z that must encode entire source sentence which includes the sentence meaning. This final state is therefore called sentence embedding1.

+ The encoder-decoder model is designed at its each step to be auto-regressive - i.e. use previously generated symbols as extra input while generating next symbol. Thus, xi+yi−1→yi
