import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


class ModelPredict(object):

    def __init__(self, model_path: str, tk_path: str):
        self.model = load_model(filepath=model_path)
        self.tk = pickle.load(open(tk_path, 'rb'))

        test_tokenized = self.tk.texts_to_sequences(["haha"])
        X = pad_sequences(test_tokenized, maxlen=50)
        pre = self.model.predict(X, batch_size=1)

    def pad_predict_sentiment(self, texts: list, max_len=50):
        test_tokenized = self.tk.texts_to_sequences(texts)
        X = pad_sequences(test_tokenized, maxlen=max_len)
        pre = self.model.predict(X, batch_size=1)

        return pre


if __name__ == '__main__':
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

