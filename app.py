from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from algoritmos.apriori import obtenerApriori
from algoritmos.metricasDistancia import *
from algoritmos.clustering import *
from algoritmos.analisisDatos import *
import base64
# from os.path import splitext

app = Flask(__name__)
CORS(app)


def getCSV(file):
    if file:
        try:
            if "csv" == str(file.filename).split(".")[-1]:
                return file
            # hola.csv.hola
        except:
            return None
    return None


@app.route('/analisis-datos', methods=['POST'])
def analisisDatosPOST():
    try:
        csvFile = getCSV(request.files['file'])
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
    if csvFile:
        try:
            Datos_Archivo = pd.read_csv(csvFile)
            bytes_obj = obtenerMapaCalor(Datos_Archivo)
            # Se envia el mapa de calor generado:
            # filename = csvFile.filename.split(".")[0]
            # Se hace un encode en la imagen con base64 string
            image_data = base64.b64encode(
                bytes_obj.getvalue()).decode('utf-8')
            columnas = Datos_Archivo.columns.tolist()
            # Return the image data in the JSON response
            return jsonify({'image_data': image_data, 'columnas': columnas})
        except Exception as e:
            print(e)
            return jsonify({"error": "Hay un problema con el archivo CSV"})
    else:
        return jsonify({'error': 'El archivo no es un CSV'})


@app.route('/asociacion', methods=['POST'])
def asociacionPOST():
    try:
        csvFile = getCSV(request.files['file'])
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
    if csvFile:
        try:
            Datos_Archivo = pd.read_csv(csvFile, header=None)
            soporteMinimo = float(request.form["soporteMinimo"])
            confianzaMinima = float(request.form["confianzaMinima"])
            elevacionMinima = float(request.form["elevacionMinima"])
        except:
            return jsonify({"error": "Faltan parametros en la petición"})
        try:
            ReglasDataFrame, nReglas, datosX, datosY = obtenerApriori(
                Datos_Archivo, soporteMinimo, confianzaMinima, elevacionMinima)
        except:
            return jsonify({"error": "Hay un problema con el archivo .csv"})
        # Se genera el CSV y se almacena en la variable reglasCSV para despues enviarlo en el .json
        reglasCSV = ReglasDataFrame.to_csv()
        return jsonify({
            "csv": reglasCSV,
            "nReglas": nReglas,
            "datosX": datosX,
            "datosY": datosY
        })
    else:
        return jsonify({'error': 'El archivo no es un CSV'})


@app.route('/matriz-distancia/<tipoDistancia>', methods=['POST'])
def matrizDistanciaPOST(tipoDistancia):
    # type puede ser: euclidean, chebyshev , cityblock (Manhattan) o minkowski
    try:
        csvFile = getCSV(request.files['file'])
        caracteristicas = request.form["seleccionCaracteristicas"]
        caracteristicasList = caracteristicas.split(',')
        seleccionCaracteristicas = []
        for i in caracteristicasList:
            seleccionCaracteristicas.append(i)
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
    if csvFile:
        Datos_Archivo = pd.read_csv(csvFile)
        try:
            MatDistancias = getMatDist(
                Datos_Archivo, tipoDistancia, seleccionCaracteristicas)
        except Exception as error:
            print(error)
            return jsonify({"error": "Hay un problema con el archivo CSV"})
        if not MatDistancias.empty:
            distanciasCSV = MatDistancias.to_csv()
            return jsonify({
                "csv": distanciasCSV,
            })
        else:
            return jsonify({'error': 'El parametro {} es incorrecto'.format(tipoDistancia)})
    else:
        return jsonify({'error': 'El archivo no es un CSV'})

# Esta peticion al final se elimino debido a que fue muy innecesaria
# @app.route('/distancia-objetos/<tipoDistancia>', methods=['POST'])
# def distanciaObjetosPOST(tipoDistancia):
#     # type puede ser: euclidean, chebyshev , cityblock (Manhattan) o minkowski
#     try:
#         csvFile = getCSV(request.files['file'])
#     except:
#         return jsonify({'error': 'No se logro leer el archivo'})
#     if csvFile:
#         try:
#             Datos_Archivo = pd.read_csv(csvFile)
#             indexObjeto1 = int(request.form["objeto1"])
#             indexObjeto2 = int(request.form["objeto2"])
#         except:
#             return jsonify({"error": "Faltan parametros en la petición"})
#         try:
#             Distancia = getDistObjetos(
#                 Datos_Archivo, tipoDistancia, indexObjeto1, indexObjeto2)
#         except:
#             return jsonify({"error": "Hay un problema con el archivo .csv"})
#         if Distancia:
#             return jsonify({
#                 "distancia": Distancia
#             })
#         else:
#             return jsonify({'error': 'El parametro {} es incorrecto'.format(tipoDistancia)})
#     else:
#         return jsonify({'error': 'El archivo no es un CSV'})


@app.route('/clustering/<tipoClustering>', methods=['POST'])
def clusteringPOST(tipoClustering):
    # type puede ser: euclidean, chebyshev , cityblock (Manhattan) o minkowski
    try:
        csvFile = getCSV(request.files['file'])
    except:
        return jsonify({'error': 'No se logro leer el archivo'})
    if csvFile:
        try:
            Datos_Archivo = pd.read_csv(csvFile)
            minClusters = int(request.form["minClusters"])
            maxClusters = int(request.form["maxClusters"])
            tipoDistancia = request.form["tipoDistancia"]
        except:
            return jsonify({"error": "Faltan parametros en la petición"})
        try:
            respuesta = getClustering(
                Datos_Archivo, tipoClustering, minClusters, maxClusters, tipoDistancia)
        except:
            return jsonify({"error": "Hay un problema con el archivo .csv"})
        return jsonify(respuesta)
    else:
        return jsonify({'error': 'El archivo no es un CSV'})


if __name__ == '__main__':
    app.run(debug=True)
