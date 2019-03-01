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
### Neural Encoder-Decoder Model
The Encoder-Decoder model aims at tackling the statistical machine translation problem of modeling the probability P(E|F) of the output E given the input F. The name “encoder-decoder” comes from  the  idea  that  the  first  neural  network  running  over F “encodes”  its information  as  a vector  of  real-valued  numbers  (the  hidden  state),  then  the  second  neural  network  used  to predict E “decodes” this information into the target sentence.

If the encoder is expressed as RNN<sup>(f)</sup>(·), the decoder is expressed as RNN<sup>(e)</sup>(·), and we have a softmax that takes RNN<sup>(e)</sup>’s hidden state at time step t and turns it into a probability, then our model is expressed as follows :

+ m<sub>t</sub><sup>(f)</sup>=M<sub>·,f<sub>t</sub></sub><sup>(f)</sup>
+ h<sub>t</sub><sup>(f)</sup>={ RNN<sup>(f)</sup>(m<sub>t</sub><sup>(f)</sup>,h<sub>t−1</sub><sup>(f)</sup>)  t≥1, 0  otherwise.
+ m<sub>t</sub><sup>(e)</sup>=M<sub>·,e<sub>t−1</sub></sub><sup>(e)</sup>
+ h<sub>t</sub><sup>(e)</sup>={ RNN<sup>(e)</sup>(m<sub>t</sub><sup>(e)</sup>,h<sup>(e)</sup><sub>t−1</sub>)  t≥1, h<sub>|F|</sub><sup>(f)</sup> otherwise.
+ p<sub>t</sub><sup>(e)</sup>= softmax(W<sub>hs</sub>h<sub>t</sub><sup>(e)</sup>+b<sub>s</sub>)

In the first two lines, we look up the embedding m<sub>t</sub><sup>(f)</sup> and calculate the encoder hidden state h<sub>t</sub><sup>(f)</sup> for the t<sup>th</sup> word in the source sequence F. We start with an empty vector h<sub>0</sub><sup>(f)</sup> = 0, and by h<sub>|F|</sub><sup>(f)</sup>, the encoder has seen all the words in the source sentence.  Thus, this hidden state should theoretically be able to encode all of the information in the source sentence.

In the decoder phase, we predict the probability of word e<sub>t</sub> at each time step.  First, we similarly look up m<sub>t</sub><sup>(e)</sup>, but this time use the previous word e<sub>t-1</sub>, as we must condition the probability of e<sub>t</sub> on the previous word, not on itself.  Then, we run the decoder to calculate h<sub>t</sub><sup>(e)</sup>.  This is very similar to the encoder step, with the important difference that h<sub>0</sub><sup>(e)</sup> is set to the final state of the encoder h<sup>(f)</sup><sub>|F|</sub>, allowing us to condition on F.  Finally, we calculate the probability p<sub>t</sub><sup>(e)</sup> by using a softmax on the hidden state h<sub>t</sub><sup>(e)</sup>.
While this model is quite simple (only 5 lines of equations), it gives us a straightforward and powerful way to model P(E|F).

### To get a deeper insight into the Transformer in a more illustrated format we read [The illustrated Transformer - by Jay Alammar](https://jalammar.github.io/illustrated-transformer/), However we were left curious with a few unanswered questions after reading it. 
### In this Blog we will attempt to answer those questions.

### What is Attention?
###### [AttentionPrimer](https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#attention-basis)

The basic idea behind the attention is that it tells us how much we are “focusing” on a particular source word at a particular time step. The encoder-decoder will only be able to access information about the first encoded  word  in  the  source  by passing  it  over |F| time  steps. The attention mechanism allows for the  source  encoding  to be accessed (in a weighted manner) through the context vector.

If H<sup>(f)</sup>, a matrix of vectors encoding each word in the input sentence F, is the output of the encoder, we calculate an attention vector α<sub>t</sub> that can be used to combine together the columns of H into a context vector c<sub>t</sub>.

c<sub>t</sub> = H<sup>(f)</sup>α<sub>t</sub>.

Attention between encoder and decoder is crucial in NMT.Attention is a function that maps the 2-element input (query, key-value pairs) to an output. The output given by the mapping function is a weighted sum of the values, where weights for each value measures how much each input key interacts with (or answers) the query. While the attention is a goal for many research, the novelty about transformer attention is that it is multi-head self-attention.
#### Basic Idea: (Bahdanau et al. 2015)
+ Encode each word in the sentence into a vector 
+ When decoding, perform a linear combination of these vectors, weighted by “attention weights” 
+ Use this combination in picking the next word

##### After reading the above explanation we had two major concerns:
#### 1)How are the attention weights obtained?
#### 2)What are Key, Query and Value?

## Calculating Attention weights
As before, the decoder’s hidden state h<sub>t</sub><sup>(e)</sup> is a fixed-length continuous vector representing the previous target words e<sub>1</sub><sup>t−1</sup> , initialized as h<sub>0</sub><sup>(e)</sup> = h<sup>(f)</sup><sub>|F|+1</sub>.  This is used to calculate a context vector c<sub>t</sub> that is used to summarize the source attentional context used in choosing target word e<sub>t</sub> , and initialized as c<sub>0</sub>=0.

First, we update the hidden state to h<sub>t</sub><sup>(e)</sup> based on the word representation and context vectors from the previous target time step 

h<sub>t</sub><sup>(e)</sup> = enc([embed(e<sub>t−1</sub>); c<sub>t−1</sub>],h<sub>t−1</sub><sup>(e)</sup>).

Based on this h<sub>t</sub><sup>(e)</sup>, we calculate an attention score a<sub>t</sub>, with each element equal to 

a<sub>t,j</sub> = attn score(h<sub>j</sub><sup>(f)</sup>,h<sub>t</sub><sup>(e)</sup>).

attn score(·) can be an arbitrary function that takes two vectors as input and outputs a score about how much we should focus on this particular input word encoding h<sub>j</sub><sup>(f)</sup>  at the time step h<sub>t</sub><sup>(e)</sup>. We then normalize this into the actual attention vector itself by taking a softmax over the scores:

α<sub>t</sub> = softmax(a<sub>t</sub>).

This  attention  vector  is  then  used  to  weight  the  encoded  representation H<sup>(f)</sup> to  create  a context vector c<sub>t</sub> for the current time step.

Following are three different attention functions:
 
##### 1)Dot product:
This  is  the  simplest  of  the  functions,  as  it  simply  calculates  the  similarity between h<sub>t</sub><sup>(e)</sup> and h<sub>j</sub><sup>(f)</sup> as measured by the dot product:

attn score(h<sub>j</sub><sup>(f)</sup>,h<sub>t</sub><sup>(e)</sup>) := h<sub>j</sub><sup>(f)ᵀ</sup>h<sub>t</sub><sup>(e)</sup>.

##### 2)Bilinear functions:
This function helps relax the restriction that the source and target embeddings must be in the same space by performing a linear transform parameterized by W<sub>a</sub> before taking the dot product:

attn score(h<sub>j</sub><sup>(f)</sup>,h<sub>t</sub><sup>(e)</sup>) := h<sub>j</sub><sup>(f)ᵀ</sup>W<sub>a</sub>h<sub>t</sub><sup>(e)</sup>.

##### 3)Multi-layer perceptrons:
This was the method employed by Bahdanau et al. 2015 in their original implementation of attention:

attn score(h<sub>j</sub><sup>(f)</sup>,h<sub>t</sub><sup>(e)</sup>) := w<sup>ᵀ</sup><sub>a2</sub>tanh(W<sub>a1</sub>[h<sub>t</sub><sup>(e)</sup> ; h<sub>j</sub><sup>(f)</sup>]),

where W<sub>a1</sub> and w<sub>a2</sub> are the weight matrix and vector of the first and second layers of the MLP respectively. 

## Key, Query and Value
In terms of encoder-decoder, the query is usually the hidden state of the decoder. Whereas key, is the hidden state of the encoder, and the corresponding value is normalized weight, representing how much attention a key gets. Output is calculated as a wighted sum – here the dot product of query and key is used to get a value.

It is assumed that queries and keys are of dk dimension and values are of dv dimension. Those dimensions are imposed by the linear projection discussed in the multi-head attention section. The input is represented by three matrices: queries’ matrix Q, keys’ matrix K and values’ matrix V.

The compatibility function (see Attention primer) is considered in terms of two, additive and multiplicative (dot-product) variants Bahdanau et al. 2014  with similar theoretical complexity.

#### Reffered from [CMU CS 11-747, Spring 2018 Graham Neubig](http://www.phontron.com/class/nn4nlp2018/schedule/attention.html)

![encoder-decoder](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/Encoder-decoder-models.png)

![calculating-attention-1](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/calculating-attention-.png)

![calculating-attention-2](https://github.com/agrawalnaman/Transformer-and-Attention/blob/master/calculating-attention-2.png)


