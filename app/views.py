import numpy as np

from flask import render_template
from flask import Blueprint
from flask import request
from flask_cors import *

from app.predict import ModelPredict
from config import model_dir, toka_path

blue = Blueprint('first', __name__, static_folder='../static', template_folder='../templates')

CORS(blue, supports_credentials=True)

multi_model_path = r"multi_model.hdf5"
lstm_model_path = r"bilstm_model.hdf5"
gru_model_path = r"bigru_model.hdf5"
cnn_model_path = r"cnn_model.hdf5"


model_check_dict = {
    "multi": ModelPredict(model_dir + multi_model_path, toka_path),
    "lstm": ModelPredict(model_dir + lstm_model_path, toka_path),
    "gru": ModelPredict(model_dir + gru_model_path, toka_path),
    "cnn": ModelPredict(model_dir + cnn_model_path, toka_path)
}


@blue.route('/', methods=['GET'])
def index():
    # return blue.send_static_file('index.html')
    return render_template('index.html')


@blue.route('/api/v1/predict', methods=['POST'])
def predict():

    json_data = request.get_json()

    text_list = [json_data["text"]]
    predter = model_check_dict[json_data["model"]]

    pres = predter.pad_predict_sentiment(text_list)
    print(pres)
    predictions = np.round(np.argmax(pres, axis=1)).astype(int)
    # pres = str()
    return str(predictions)




