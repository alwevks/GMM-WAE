import json
import numpy as np
import pandas as pd
import gensim
import re
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def calculate_bleu_scores(references, hypotheses):
    """
    Calculates BLEU 1-4 scores based on NLTK functionality

    Args:
        references: List of reference sentences
        hypotheses: List of generated sentences

    Returns:
        bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores

    """
    bleu_1 = np.round(100 * corpus_bleu(references, hypotheses, weights=(1.0, 0., 0., 0.)), decimals=2)
    bleu_2 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0., 0.)), decimals=2)
    bleu_3 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.34, 0.33, 0.33, 0.)), decimals=2)
    bleu_4 = np.round(100 * corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)), decimals=2)
    return bleu_1, bleu_2, bleu_3, bleu_4


def calculate_ngram_diversity(corpus):
    """
    Calculates unigram and bigram diversity

    Args:
        corpus: tokenized list of sentences sampled

    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score

    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity


def calculate_entropy(corpus):
    """
    Calculates diversity in terms of entropy (using unigram probability)

    Args:
        corpus: tokenized list of sentences sampled

    Returns:
        ent: entropy on the sample sentence list

    """
    fdist = FreqDist(corpus)
    total_len = len(corpus)
    ent = 0
    for k, v in fdist.items():
        p = v / total_len

        ent += -p * np.log(p)

    return ent


def tokenize_sequence(sentences, filters, max_num_words, max_vocab_size):
    """
    Tokenizes a given input sequence of words.

    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary

    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence

    """

    sentences = [' '.join(word_tokenize(s)[:max_num_words]) for s in sentences]

    tokenizer_filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(filters=tokenizer_filters)
    tokenizer.fit_on_texts(sentences)

    word_index = dict()
    word_index['PAD'] = 0
    word_index['UNK'] = 1
    word_index['GO'] = 2
    word_index['EOS'] = 3

    for i, word in enumerate(dict(tokenizer.word_index).keys()):
        word_index[word] = i + 4

    tokenizer.word_index = word_index
    x = tokenizer.texts_to_sequences(list(sentences))

    for i, seq in enumerate(x):
        if any(t >= max_vocab_size for t in seq):
            seq = [t if t < max_vocab_size else word_index['UNK'] for t in seq]
        seq.append(word_index['EOS'])
        x[i] = seq

    x = pad_sequences(x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['PAD'])

    word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}

    return x, word_index


def create_embedding_matrix(word_index, embedding_dim, w2v_path):
    """
    Create the initial embedding matrix for TF Graph.

    Args:
        word_index: dictionary storing the word-to-index correspondence
        embedding_dim: word2vec dimension
        w2v_path: file path to the w2v pickle file

    Returns:
        embeddings_matrix : numpy 2d-array with word vectors

    """
    w2v_model = gensim.models.Word2Vec.load(w2v_path)
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index), embedding_dim))
    for word, i in word_index.items():
        try:
            embeddings_vector = w2v_model[word]
            embeddings_matrix[i] = embeddings_vector
        except KeyError:
            pass

    return embeddings_matrix


def get_sentences(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    return data


def clean_sentence(sent):
    sent = re.sub(r'[^\w\s\?\.\,]', '', sent.strip().lower())  # Lower case, remove punctuations (except , ? .)
    sent = re.sub(r'(([a-z]*)\d+.?\d*\%?)', ' NUM ', sent.strip())  # Replace Numbers with <NUM> token
    return sent


def get_batches(x, labels, batch_size):
    """
    Generate inputs and targets in a batch-wise fashion for feed-dict

    Args:
        x: entire source sequence array
        batch_size: batch size

    Returns:
        x_batch, y_batch, sentence_length

    """

    for batch_i in range(0, len(x) // batch_size):
        start_i = batch_i * batch_size
        x_batch = x[start_i:start_i + batch_size]
        y_batch = x[start_i:start_i + batch_size]
        labels_batch = labels[start_i:start_i + batch_size]

        sentence_length = [np.count_nonzero(seq) for seq in x_batch]

        yield x_batch, y_batch, labels_batch, sentence_length

def _parse_function(x):
    return x, x, tf.count_nonzero(x)

def get_dataset(x):
    dataset = tf.data.Dataset.from_tensor_slices((x))
    dataset = dataset.cache()
    dataset = dataset.shuffle(30000)
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def plot_2d(zvectors, labels, method):
    if method == 'tsne':
        cluster = TSNE(n_components=2, random_state=17)
    else:  # PCA
        cluster = PCA(n_components=2, random_state=17)

    cluster_result = cluster.fit_transform(X=zvectors)
    labels = labels[:cluster_result.shape[0]]
    labels = np.array(labels)

    class_dict = {0: 'automobile', 1: 'home and kitchen'}
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(w=10, h=10)
    ax.scatter(cluster_result[np.where(labels == 0), 0], cluster_result[np.where(labels == 0), 1], s=6,
               label=class_dict[0])
    ax.scatter(cluster_result[np.where(labels == 1), 0], cluster_result[np.where(labels == 1), 1], s=6,
               label=class_dict[1])
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()

def get_mnli_data(path):
    df = pd.DataFrame(columns=['sentence', 'genre'])
    sentences = []
    genres = []
    count = {'travel': 0, 'telephone': 0, 'fiction': 0, 'government': 0}
    for i, line in enumerate(open(path)):
        data = json.loads(line)
        if data['genre'] == 'slate' or count[data['genre']] > 10240:
            continue
        genres.append(data['genre'])
        count[genres[-1]] += 1
        sentences.append(data['sentence1'])

    data = {'sentence': sentences, 'genre': genres}
    data = pd.DataFrame(data)
    sentences = []
    genres = []
    idx_genre = {}
    for i, genre in enumerate(set(data.genre)):
        length = (len(data[data.genre == genre]) // 512) * 512
        sentences.extend(data[data.genre == genre].sentence.values[:length])
        genres.extend([i] * length)
        idx_genre[i] = genre

    # Shuffling the data into 512 same-genre batches
    sentences = np.reshape(sentences, (-1, 512))
    genres = np.reshape(genres, (-1, 512))
    sentences, genres = shuffle(sentences, genres)
    sentences = list(np.reshape(sentences, (-1)))
    genres = list(np.reshape(genres, (-1)))
    print(idx_genre)
    return sentences, genres, idx_genre

def new_get_mnli_data(genre_path, sentence_path):
    df = pd.DataFrame(columns=['sentence', 'genre'])
    sentences = []
    genres = []
    count = {'travel': 0, 'telephone': 0, 'fiction': 0, 'government': 0}
    for i, genre in enumerate(open(genre_path)):
        genre = genre.strip()
        genres.append(genre)
        count[genre] += 1

    for i, line in enumerate(open(sentence_path)):
        line = line.strip()
        sentences.append(line)

    sentences_ = []
    genres_ = []
    count = {'travel': 0, 'telephone': 0, 'fiction': 0, 'government': 0}
    for i in range(len(genres)):
        if count[genres[i]] > 12800:
            continue
        genres_.append(genres[i])
        sentences_.append(sentences[i])
        count[genres[i]] += 1
    sentences = sentences_
    genres = genres_

    data = {'sentence': sentences, 'genre': genres}
    data = pd.DataFrame(data)
    sentences = []
    genres = []
    idx_genre = {}
    for i, genre in enumerate(set(data.genre)):
        length = (len(data[data.genre == genre]) // 512) * 512
        sentences.extend(data[data.genre == genre].sentence.values[:length])
        genres.extend([i] * length)
        idx_genre[i] = genre

    # Shuffling the data into 512 same-genre batches
    sentences = np.reshape(sentences, (-1, 512))
    genres = np.reshape(genres, (-1, 512))
    sentences, genres = shuffle(sentences, genres)
    sentences = list(np.reshape(sentences, (-1)))
    genres = list(np.reshape(genres, (-1)))
    print(idx_genre)
    return sentences, genres, idx_genre

