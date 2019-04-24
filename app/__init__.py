from flask import Flask

from app.views import blue
from flask_cors import *


def create_app():
    app = Flask(__name__, static_folder='../static', template_folder='../templates')
    CORS(blue, supports_credentials=True)
    app.register_blueprint(blueprint=blue)
    # 跨域
    CORS(app, supports_credentials=True)

    return app


