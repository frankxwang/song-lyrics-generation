# import gensim.models.word2vec as w2v
# import re
import keras
import numpy as np
import pickle
import nltk
from nltk.tokenize.moses import MosesDetokenizer
max_words = 29
length = 100

print("Loading Word2Vec")
with open("data/song2vec.pkl", "rb") as f:
    song2vec = pickle.load(f)

print("Loading Model")
model = keras.models.load_model("data/trained.hdf5")

print("Loading Lexicon")
with open("data/lexicon.pkl", 'rb') as f:
    lexicon = pickle.load(f)
    lexicon = {v: k for k, v in lexicon.items()}

detokenizer = MosesDetokenizer()

while True:
    words = input("Type in the beginning of your song (" + str(max_words) + " words maximum) \n")
    if words == "makeitrandom":
        x = 6 * np.random.random((1, length)) - 3
        x = np.append(np.zeros((max_words - 1, length)), x, axis=0)
        song = np.array([])
    else:
        song = nltk.word_tokenize(words.lower())[:max_words]
        print("Processing")
        x = np.zeros((max_words - len(song), length))
        try:
            for word in song:
                x = np.append(x, np.array([song2vec[word]]), axis=0)
        except KeyError:
            print("It looks like you have typed something wrong, please check your spelling and try again.")
            continue
    print("Generating")
    output = song
    for i in range(length):
        y = model.predict(np.array([x]))[0]
        y = lexicon[np.argmax(y)]
        output = np.append(output, y)
        y = np.array([song2vec[y]])
        x = x[1:]
        x = np.append(x, y, axis=0)
    print(detokenizer.detokenize(output, return_str=True))
