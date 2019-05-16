import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from app.predict import ModelPredict


def test_class_m():
    model_path = r"C:\code\python3workspace\sentiment_demo\data\model\pres\gpu\bilstm_model.hdf5"
    tk_path = r"C:\code\python3workspace\sentiment_demo\data\model\toka.bin"
    mp = ModelPredict(model_path, tk_path)
    pre = mp.pad_predict_sentiment(["unhappy"], 50)
    print(pre)


def test_model():
    texts = ["happy", "very happy", "sad", "unhappy"]
    max_len = 50
    tk_path = r"C:\code\python3workspace\sentiment_demo\data\model\toka.bin"
    model_path = r"C:\code\python3workspace\sentiment_demo\data\model\pres\gpu\multi_model429.hdf5"
    tk = pickle.load(open(tk_path, 'rb'))
    model = load_model(filepath=model_path)
    test_tokenized = tk.texts_to_sequences(texts)
    X = pad_sequences(test_tokenized, maxlen=max_len)
    pre = model.predict(X, batch_size=1024)

    print(pre)


if __name__ == '__main__':
    test_model()
    # test_class_m()
