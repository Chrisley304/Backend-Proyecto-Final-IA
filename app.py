from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from algoritmos.apriori import obtenerApriori
from algoritmos.metricasDistancia import *
# from os.path import splitext

app = Flask(__name__)
CORS(app)
def getCSV(file):
    if file:
        try:
            if "csv" ==  str(file.filename).split(".")[-1]: return file
            # hola.csv.hola
        except:
            return None
    return None

@app.route('/asociacion', methods=['POST'])
def asociacionPOST():
    try:
        csvFile = getCSV(request.files['file'])
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
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


@app.route('/metricas-distancia/<tipoDistancia>', methods=['POST'])
def metricasDistanciaPOST(tipoDistancia):
    # type puede ser: euclidean, chebyshev , cityblock (Manhattan) o minkowski
    try:
        csvFile = getCSV(request.files['file'])
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
    if csvFile:
        Datos_Archivo = pd.read_csv(csvFile)
        MatDistancias = getMatDist(Datos_Archivo, tipoDistancia)
        if not MatDistancias.empty:
            distanciasCSV = MatDistancias.to_csv()
            return jsonify({
                "csv": distanciasCSV,
                "prueba": "prueba"
            })
        else:
            return jsonify({'error': 'El parametro {} es incorrecto'.format(tipoDistancia)})
    else:
        return jsonify({'error': 'El archivo no es un CSV'})

""" Otras formas de interactuar con la api
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
"""


if __name__ == '__main__':
    app.run(debug=True)
