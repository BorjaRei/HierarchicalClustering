import os
import json
import gzip
from time import time
from nltk.corpus.reader.chasen import test
import pandas as pd
from urllib.request import urlopen

import random
import numpy as np
from collections import defaultdict

# CARGAMOS LOS DATOS.

dfMergedfMeta = []

dfMergedfMeta = pd.read_csv('intuit.csv', low_memory=False)

# SACAMOS LOS 10 PRIMEROS PARA COMPROBAR QUE FUNCIONA
print(dfMergedfMeta[:10])

# INDICES
print(dfMergedfMeta.keys())

# ANALISIS DEL CONJUNTO DE DATOS

print(dfMergedfMeta.info())  # Numero y Tipo de atributos
print("---------------------------------------------------")

print(dfMergedfMeta.describe())  # Min,Max,Media,Desviacion ...

print("---------------------------------------------------")

print(dfMergedfMeta.duplicated().describe())  # Instacias repetidas
dfMergedfMeta.drop_duplicates()
print("---------------------------------------------------")

print("MISSING VALUES")
print(dfMergedfMeta.isnull().sum())  # Missing values
dfMergedfMeta.dropna().reset_index(drop=True)
print(dfMergedfMeta.isnull().sum())

# FILTROS ENCODING/TOKENIZATION/RE-CASING

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import csv
import numpy

i = 0
vocabComplete = []
sentences = []
cachedStopWords = stopwords.words('english')

df = dfMergedfMeta['reviewText']

for reg in df:
    print("---------------------------------------------------")
    min_length = 3
    words = word_tokenize(str(reg));

    words = [word for word in words
             if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)));

    p = re.compile("[a-z]");

    filtered_tokens = list(filter(lambda token:
                                  p.match(token) and len(token) >= min_length, tokens));

    vocabComplete = vocabComplete + filtered_tokens

    print("---------------------------------------------------")
    csvsalida = open('salida.txt', 'w', newline='')
    csvsalida.write("---------------------------------------------------------------\n")
    csvsalida.write(str(filtered_tokens))

    print(filtered_tokens)
    sentences.append(filtered_tokens)  ##metemos las instancias limpiadas en una lista para el BoW

# REPRESENTACION BoW
"""
print(sentences)
sen = 0
# Se recorren las intsancias y por cada palabra de la instancia (limpiada anteriormente) se mira cuntas veces aparece
# en el vocabulario
while sen < len(sentences):
    words = sentences[sen]
    bag_vector = numpy.zeros(len(vocabComplete))
    for w in words:
        for i, word in enumerate(vocabComplete):
            if word == w:
                bag_vector[i] += 1
    sen = sen + 1
    print("{0}\n{1}\n".format(words, numpy.array(bag_vector)))
"""
##vocabCompleteUnico = pd.unique(vocabComplete)  ##quitar repetidos

##VOCABULARIO CON INDICES // Representacion numerica / https://www.tensorflow.org/tutorials/text/word2vec

vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in vocabComplete:
    if token not in vocab:
        vocab[token] = index
        index += 1
vocab_size = len(vocab)
print(vocab)
print("---------")
print(vocab_size)
print("---------")
inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

vector = []
for sentence in sentences:
    vectorAux = []
    for word in sentence:
        vectorAux.append(vocab[word])
    vector.append(vectorAux)

print(sentences[3])
print(vector[3])
print(len(vector[3]))
print(len(sentences[3]))
print("-------------------")
print(len(sentences))
print(len(vector))

##GENSIM Word2Vec Word Embedding (https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial)

print("----------------------------------------------------")

##Palabaras mas frecuentes
word_freq = defaultdict(int)
for word in vocabComplete:
    word_freq[word] += 1
print(len(word_freq))

print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])  ## PALABRAS MAS REPETIDAS
"""
Why I seperate the training of the model in 3 steps:
I prefer to separate the training in 3 distinctive steps for clarity and monitoring.

Word2Vec():
In this first step, I set up the parameters of the model one-by-one.
I do not supply the parameter sentences, and therefore leave the model uninitialized, purposefully.

.build_vocab():
Here it builds the vocabulary from a sequence of sentences and thus initialized the model.
With the loggings, I can follow the progress and even more important, the effect of min_count and sample on the word corpus. I noticed that these two parameters, and in particular sample, have a great influence over the performance of a model. Displaying both allows for a more accurate and an easier management of their influence.

.train():
Finally, trains the model.
The loggings here are mainly useful for monitoring, making sure that no threads are executed instantaneously.
"""

from gensim.models import Word2Vec
import multiprocessing

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=50,
                     window=2,
                     vector_size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores - 1)  # Creando el modelo

w2v_model.build_vocab(sentences, progress_per=10000)  ##Creando el vocabulario

print(w2v_model)

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)  ##Training
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

print("---------------------------------------------------------------------")
print(w2v_model.wv.most_similar(positive=["intuit"]))  ## Palabras relacionados con ...
print("---------------------------------------------------------------------")

print(w2v_model.wv.similarity('intuit', 'tax'))  ##Porcentaje de similarity
print("---------------------------------------------------------------------")

print(w2v_model.wv.doesnt_match(['intuit', 'tax', 'data']))  # Cual sobra

print("---------------------------------------------------------------------")

from sklearn.decomposition import PCA
from matplotlib import pyplot

##REPRESENTACION 1

words = list(w2v_model.wv.index_to_key)
print(words)
X = w2v_model.wv.__getitem__(w2v_model.wv.index_to_key)  ##VERSION DESACTUALIZADA
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()

##REPRESENTACION 2

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)

    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                  }
                     )

    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
        p1.text(df["x"][line],
                df['y'][line],
                '  ' + df["words"][line].title(),
                horizontalalignment='left',
                verticalalignment='bottom', size='medium',
                color=df['color'][line],
                weight='normal'
                ).set_size(15)

    plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
    plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

    plt.title('t-SNE visualization for {}'.format(word.title()))


##To make the visualizations more relevant, we will look at the relationships between a query word (in **red**),
##its most similar words in the model (in **blue**), and other words from the vocabulary (in **green**).

# 10 Most similar words vs. 8 Random words

##print(tsnescatterplot(w2v_model, 'tax', ['option', 'offer', 'help']))

# 10 Most similar words vs. 10 Most dissimilar

##print(tsnescatterplot(w2v_model, 'tax', [i[0] for i in w2v_model.wv.most_similar(negative=["tax"])]))

##REPRESENTACION 3
analizeWord = 'intuit'
test = [i[0] for i in w2v_model.wv.most_similar(positive=[analizeWord])]
test.append(analizeWord)
words = list(test)
print(words)
X = w2v_model.wv.__getitem__(test)
print(X)
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()

# 10 Most similar words vs. 8 Random words

##print(tsnescatterplot(w2v_model, 'tax', ['option', 'offer', 'help']))

# 10 Most similar words vs. 10 Most dissimilar

##print(tsnescatterplot(w2v_model, 'tax', [i[0] for i in w2v_model.wv.most_similar(negative=["tax"])]))











