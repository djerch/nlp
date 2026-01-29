# Imports necessary modules to embed words in text files.
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from threading import Thread
import fileinput

# Reads previous scraped text file created for the account CBCNews.

corpus = ""

count = 0

for line in fileinput.input(['CBCNewsNov142022toNov142011.txt']):
    line.replace('Â', '')

f = corpus.replace('\n', ' ')

sentences = []

# iterate through each sentence in the file.
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words.
    for j in word_tokenize(i):
            temp.append(j.lower())
            print(j.lower())

    sentences.append(temp)
    print(temp)

# Creates the model.
# Note: to implement Skip Gram model, simply change sg = 1.
# CBOW is better for larger corpus and more frequent words.
model = Word2Vec(sentences, min_count = 3, sg = 0)
model.save('CBCNewsNov142022toNov142011_Trained_Model.p')