import pickle

import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SpatialDropout2D
from keras.models import Model, load_model
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
# embedding_path = "d:/data/word2vec/en/crawl-300d-2M.vec/crawl-300d-2M.vec"
embedding_path = "C:/data/word2vec/en/deps.words/deps.words"

root_path = r"C:\code\python3workspace\sentiment_demo\data\movie-review-sentiment-analysis-kernels-only"

log_filepath = r"C:\code\python3workspace\sentiment_demo\model\log"

toka_path = r"C:\code\python3workspace\sentiment_demo\data\model\toka.bin"

model_path = r"C:\code\python3workspace\sentiment_demo\data\model\pres\cpu\bigru_model.hdf5"

train = pd.read_csv(root_path + r'\train.tsv', sep="\t")
test = pd.read_csv(root_path + r'\test.tsv', sep="\t")
sub = pd.read_csv(root_path + r'\sampleSubmission.csv', sep=",")

train.loc[train.SentenceId == 2]

print('Average count of phrases per sentence in train is {0:.0f}.'.format(
    train.groupby('SentenceId')['Phrase'].count().mean()))
print('Average count of phrases per sentence in test is {0:.0f}.'.format(
    test.groupby('SentenceId')['Phrase'].count().mean()))

print('Number of phrases in train: {}. Number of sentences in train: {}.'.format(train.shape[0],
                                                                                 len(train.SentenceId.unique())))
print('Number of phrases in test: {}. Number of sentences in test: {}.'.format(test.shape[0],
                                                                               len(test.SentenceId.unique())))

print('Average word length of phrases in train is {0:.0f}.'.format(
    np.mean(train['Phrase'].apply(lambda x: len(x.split())))))
print('Average word length of phrases in test is {0:.0f}.'.format(
    np.mean(test['Phrase'].apply(lambda x: len(x.split())))))

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

y = train['Sentiment']

tk = Tokenizer(lower=True, filters='')
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train['Phrase'])
test_tokenized = tk.texts_to_sequences(test['Phrase'])

pickle.dump(tk, open(toka_path, 'wb'))

max_len = 50
X_train = pad_sequences(train_tokenized, maxlen=max_len)
X_test = pad_sequences(test_tokenized, maxlen=max_len)
embed_size = 300
max_features = 20000


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)


def build_model(lr=0.0, lr_d=0.0, units=0, dr=0.0):
    inp = Input(shape=(max_len,))
    x = Embedding(19479, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(GRU(units, return_sequences=True))(x1)
    # x_gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(x1)
    avg_pool1_gru = GlobalAveragePooling1D()(x_gru)
    max_pool1_gru = GlobalMaxPooling1D()(x_gru)

    x = concatenate([avg_pool1_gru, max_pool1_gru])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100, activation='relu')(x))
    x = Dense(5, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(X_train, y_ohe, batch_size=32, epochs=20, validation_split=0.1, verbose=1,
              callbacks=[check_point, early_stop, tb_cb])
    model = load_model(model_path)
    return model


model = build_model(lr=1e-4, lr_d=0, units=128, dr=0.5)
pred = model.predict(X_test, batch_size=1024)

predictions = np.round(np.argmax(pred, axis=1)).astype(int)
# for blending if necessary.
# (ovr.predict(test_vectorized) + svc.predict(test_vectorized) + np.round(np.argmax(pred, axis=1)).astype(int)) / 3
sub['Sentiment'] = predictions
sub.to_csv("blend.csv", index=False)
