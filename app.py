from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from algoritmos.apriori import *
from os.path import splitext

app = Flask(__name__)
CORS(app)
def getCSV(file):
    if file:
        try:
            if ".csv" in str(file.filename): return file
        except:
            return None
    return None

@app.route('/asociacion', methods=['POST'])
def asociacionPOST():
    csvFile = getCSV(request.files['file'])
    if csvFile:
        Datos_Archivo = pd.read_csv(csvFile, header=None)
        soporteMinimo = float(request.form["soporteMinimo"])
        confianzaMinima = float(request.form["confianzaMinima"])
        elevacionMinima = float(request.form["elevacionMinima"])
        listaResultados = obtenerApriori(Datos_Archivo,soporteMinimo,confianzaMinima,elevacionMinima)
        ReglasDataFrame = pd.DataFrame(listaResultados)
        # Se genera el CSV y se almacena en la variable reglasCSV para despues enviarlo en el .json
        reglasCSV = ReglasDataFrame.to_csv()
        return jsonify({
            "csv": reglasCSV,
            "nReglas": len(listaResultados),
        })
    else:
        return jsonify({'error': 'El archivo no es un CSV'})


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
