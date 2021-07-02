# -*- coding: UTF-8 -*-
import app.model as model
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def getResult():
    text = "點進去好麻煩"
    result = model.predict(text)
    return jsonify({'result': str(result)})

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    input=insertValues['inputText']
    print(input)
    # 進行預測
    result = model.predict(input)

    return jsonify({'result': str(result)})