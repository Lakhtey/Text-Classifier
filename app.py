from selector.ModelSelector import AvailableMLModels
from selector.VectorSelector import AvailableVectorTypes
from Executor import predict, train
import json
from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'
  
@app.route('/greet')
def say_hello():
  return 'Hello from Server'

@app.route('/allmodels')
def getAllAvailableModels():
    dict = {}
    for s in AvailableMLModels:
        dict[s.name] = s.value
    return json.dumps(dict)

@app.route('/allvectorTypes')
def getAllAvailableVectorTypes():
    dict = {}
    for s in AvailableVectorTypes:
        dict[s.name] = s.value
    return json.dumps(dict)

@app.route('/predict', methods=['POST'])
def getprediction():
    # payload = json.loads(request.get_json())
    payload = request.get_json()
    return json.dumps(predict(AvailableMLModels(payload["model"]),AvailableVectorTypes(payload["vectortype"]),payload["filepath"]))

@app.route('/train', methods=['POST'])
def train_model():
    # payload = json.loads(request.get_json())
    payload = request.get_json()
    return json.dumps(train(AvailableMLModels(payload["model"]),AvailableVectorTypes(payload["vectortype"])))