"""
Created on Tue Jul 26 14:17:49 2022

@author: Dylan
"""

# Imports necessary modules to embed words in text files. 
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Creates a trained model based off scraped tweets from given Twitter account.
def create_model():
    # Reads previous scraped text file created for the account CBCIndigenous.
    corpus = open('CBCIndigenous.txt', encoding = 'utf-8')
    c = corpus.read()

    # Replaces escape characters with a space in order to create a huge one liner.
    f = c.replace('\n', ' ')
    corpus.close()

    sentences = []

    # iterate through each sentence in the file.
    for i in sent_tokenize(f):
        temp = []
    
        # tokenize the sentence into words.
        for j in word_tokenize(i):
                temp.append(j.lower())
        
        sentences.append(temp)
    
    # Creates the model.
    # Note: to implement Skip Gram model, simply change sg = 1. 
    # CBOW is better for larger corpus and more frequent words.
    model = Word2Vec(sentences = sentences, min_count = 3, sg = 0)
    return model


def scatter_plot(model):
    target_words = ['security', 'fairness', 'safety', 'workplace', 'employment', 'identity']
    color_list = ['green'] * len(target_words)
    
    # Creates an empty list which will contain vectors of information related to the words (a 2D-array).
    X = []
    
    # Creates a new list containing all labels for each point.
    labels = []
    
    # Adds all target words vector information and label into respective lists.
    for word in target_words:
        X.append(model.wv.__getitem__(word))
        labels.append(word)
    
    # adds all words from three most similar words to labels.
    for word in target_words:
        similar_words = model.wv.most_similar(word, topn = 3)
        print(word)
        print(similar_words)
        for wrd in similar_words:
            X.append(model.wv.__getitem__(wrd[0]))
            labels.append(wrd[0])
            color_list.append('lightgreen')
            
    # transforms the multi-dimensional vector information into a 2D vector.
    vectors = PCA(n_components = 2).fit_transform(X)
    x = vectors[:, 0]
    y = vectors[:, 1]
    #print(labels)
    
    # labels all points with the targeted words and three most similar associated words.
    for i in list(range(len(labels))):
        plt.scatter(x[i], y[i], c = color_list[i])
        plt.annotate(labels[i], (x[i], y[i]), ha = 'center')
        
    plt.show()

model = Word2Vec.load('CBCIndigenous_Trained_Model.p')
scatter_plot(model)
