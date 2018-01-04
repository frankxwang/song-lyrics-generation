import multiprocessing
import os
import pickle
import re
from collections import deque

import gensim.models.word2vec as w2v
import pandas as pd

from model import make_model

size = 100
window = 7
min_count = 1
max_words = 30
max_data = 10000
sample_per_data = 20


def window(seq, n=max_words, samples=sample_per_data):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield list(win)
    append = win.append
    i = 0
    for e in it:
        if i == samples:
            break
        i += 1
        append(e)
        yield list(win)
data = pd.read_csv("/data/songdata.csv")
data = data[data.artist != 'Lata Mangeshkar']
songs = []
for song in data["text"][:max_data]:
    song = re.sub("[^A-Za-z]", " ", song).lower().split()
    # songs.append(song[:max_words])
    windows = list(window(song))
    for section in windows:
        songs.append(section)
print(len(songs))

if not os.path.isfile("/data/song2vec.w2v"):
    model = w2v.Word2Vec(songs, size=size, window=window, min_count=min_count, workers=multiprocessing.cpu_count())
    model.save("song2vec.w2v")
else:
    model = w2v.Word2Vec.load("/data/song2vec.w2v")


def word2dictionary(dat):
    # iterate thru all words and assign id
    w_dict = {}
    count = 0
    for song in dat:
        for word in song:
            if word not in w_dict:
                w_dict[word] = count
                count += 1
    return w_dict

if not os.path.isfile("/data/lexicon.pkl"):
    lexicon = word2dictionary(songs)
    with open("/data/lexicon.pkl", 'wb') as f:
        pickle.dump(lexicon, f, pickle.HIGHEST_PROTOCOL)
else:
    with open("/data/lexicon.pkl", 'rb') as f:
        lexicon = pickle.load(f)
# if not os.path.isfile("x.npy") or not os.path.isfile("y.npy"):
# print("do")
# inputs = [[[0] * size] * max_words] * 640000
# print("done")
inputs = []
outputs = []
index = 0
for song in songs:
    if max_words - len(song) > 0:
        song_vec = [[0] * size] * (max_words - len(song)) + model[song].tolist()
    else:
        song_vec = model[song].tolist()
    inputs.append(song_vec[:-1])
    # one_hot = [[0] * len(lexicon)] * max_words
    # # print(one_hot)
    # for i in range(len(song)):
    #     one_hot[i + max_words - len(song)][lexicon[song[i]]] = 1
    output = [lexicon[song[len(song) - 1]]]
    outputs.append(output)
    if index % 10000 == 0:
        print("Song Number: " + str(index))
    index += 1
# np.save("x.npy", np.array(inputs))
# np.save("y.npy", np.array(outputs))
x = inputs
y = outputs
# else:
#     x = np.load("x.npy")[:-1]
#     y = np.load("y.npy")[1:]

print("Making model")
model = make_model(x, y, size, len(lexicon), max_words - 1)
model.save("/output/trained.hdf5")
