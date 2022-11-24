from flask import Flask, request, jsonify
from flask_cors import CORS
import algoritmos

app = Flask(__name__)
CORS(app)

@app.route('/users', methods=['POST'])
def createUser():
    print(request.json)
    respuesta = "El usuario ingresado es: {} {} con el username: {}".format(
        request.json['nombre'], request.json['apellido'], request.json['username'])
    print(respuesta)
    return respuesta


@app.route('/users', methods=['GET'])
def getUsers():
    return 'getUsers'


@app.route('/user/<id>', methods=['GET'])
def getUser(id):
    return 'getUser: ' + id


@app.route('/user/<id>', methods=['DELETE'])
def deleteUser(id):
    return 'deleteUser'


@app.route('/user/<id>', methods=['PUT'])
def updateUser(id):
    return 'updateUser'


if __name__ == '__main__':
    app.run(debug=True)
