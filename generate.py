import gensim.models.word2vec as w2v
import re
import keras
import numpy as np
import pickle
max_words = 29
length = 100

print("Loading Word2Vec")
song2vec = w2v.Word2Vec.load("data/song2vec.w2v")

print("Loading Model")
model = keras.models.load_model("trained.hdf5")

print("Loading Lexicon")
with open("data/lexicon.pkl", 'rb') as f:
    lexicon = pickle.load(f)
    lexicon = {v: k for k, v in lexicon.items()}

while True:
    words = input("Type in the beginning of your song (" + str(max_words) + " words maximum) \n")
    song = re.sub("[^A-Za-z]", " ", words).lower().split()[:max_words]
    print("Processing")
    x = np.zeros((max_words - len(song), length))
    try:
        for word in song:
            x = np.append(x, [song2vec[word]], axis=0)
    except:
        print("It looks like you have typed something wrong, please check your spelling and try again.")
        continue
    print("Generating")
    output = " ".join(song)
    for i in range(length):
        y = model.predict(np.array([x]))[0]
        y = lexicon[np.argmax(y)]
        output += " " + y
        y = np.array([song2vec[y]])
        x = x[1:]
        x = np.append(x, y, axis=0)
    print(output)
