from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adagrad

# num_hidden = 300
dropout = 0.1
# num_layers = 2
batch_size = 32
epochs = 2
learning_rate = 0.05


def make_model(x, y, num_features, out_size, batch_len):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(batch_len, num_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(out_size))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adagrad(lr=learning_rate))
    model.fit(x, y, batch_size=batch_size, epochs=epochs)

    return model