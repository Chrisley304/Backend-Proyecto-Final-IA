import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn import model_selection
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier


def getArbolYBosque(Datos: DataFrame, seleccionCaracteristicas: list, tamanioMuestra, variableClase, filename, profundidadMaxima: int):
    # Se obtienen los valores de la variable clase
    valoresVariableClase = list(Datos[variableClase].unique())
    # Se verifica que solo existan 2 clases
    if len(valoresVariableClase) > 2:
        return {"error": "La variable predictora tiene mas de 2 clases, el algoritmo solo recibe variables predictorias binarias."}
    else:
        # Se verifica que las clases sean 0 y 1
        if not 0 in valoresVariableClase or not 1 in valoresVariableClase:
            # Se convierten los valores de la variable predictora a 0 y 1 en el caso que no tengan este formato
            if len(valoresVariableClase) == 2:
                Datos[variableClase] = Datos[variableClase].replace(
                    valoresVariableClase[0], 0)
                Datos[variableClase] = Datos[variableClase].replace(
                    valoresVariableClase[1], 1)
            else:
                return {"error": "La variable predictora no tiene clases, el algoritmo solo recibe variables predictorias binarias."}

    variablesPredictoras = np.array(Datos[seleccionCaracteristicas])
    Y = np.array(Datos[[variableClase]])
    # Con los datos listos, se comienza a aplicar el algoritmo:
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(variablesPredictoras, Y,
                                                                                    test_size=tamanioMuestra,
                                                                                    random_state=0,
                                                                                    shuffle=True)

    # Se crea el arbol de decision
    MatrizClasificacionArbolDesc, exactitudArbolDesc, graficaArbolDesc, ClasificacionAD = crearArbolDecision(X_train, X_validation, Y_validation,
                                                                                                             Y_train, profundidadMaxima, seleccionCaracteristicas)
    # Se crea el bosque aleatorio
    MatrizClasificacionBosqueAleatorio, exactitudBosqueAleatorio, ClasificacionBA = crearBosqueAleatorio(X_train, X_validation, Y_validation,
                                                                                                         Y_train, int(profundidadMaxima/2), seleccionCaracteristicas)
    # Se crea la gráfica ROC
    datos_graficaROC = crearGraficaROC(
        ClasificacionAD, ClasificacionBA, X_validation, Y_validation,)
    return {"matrizClasificacionArbolDescision": MatrizClasificacionArbolDesc.to_csv(), "exactitudPromedioArbolDesc": exactitudArbolDesc, "graficaArbolDesc": graficaArbolDesc, "matrizClasificacionBosqueAleatorio": MatrizClasificacionBosqueAleatorio.to_csv(), "exactitudPromedioBosqueAleatorio": exactitudBosqueAleatorio, "graficaROC": datos_graficaROC}


def crearArbolDecision(X_train, X_validation, Y_validation, Y_train, profundidadMaxima, seleccionCaracteristicas):
    # Se entrena el modelo a partir de los datos de entrada
    ClasificacionAD = DecisionTreeClassifier(max_depth=profundidadMaxima,
                                             min_samples_split=4,
                                             min_samples_leaf=2,
                                             random_state=0)
    ClasificacionAD.fit(X_train, Y_train)
    # Clasificación final
    Y_ClasificacionAD = ClasificacionAD.predict(X_validation)
    exactitudAlgoritmo = accuracy_score(Y_validation, Y_ClasificacionAD)
    # Matriz de clasificación
    ModeloClasificacion1 = ClasificacionAD.predict(X_validation)
    Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(),
                                        ModeloClasificacion1,
                                        rownames=['Reales'],
                                        colnames=['Clasificación'])
    # Se crea la gráfica del árbol de decisión
    graficaArbol = crearGraficaArbol(ClasificacionAD, seleccionCaracteristicas)
    return Matriz_Clasificacion1, exactitudAlgoritmo, graficaArbol, ClasificacionAD


def crearBosqueAleatorio(X_train, X_validation, Y_validation, Y_train, profundidadMaxima, seleccionCaracteristicas):
    # Se entrena el modelo a partir de los datos de entrada
    ClasificacionBA = RandomForestClassifier(n_estimators=105,
                                             max_depth=profundidadMaxima,
                                             min_samples_split=4,
                                             min_samples_leaf=2,
                                             random_state=1234)
    ClasificacionBA.fit(X_train, Y_train)
    # Clasificación final
    Y_ClasificacionBA = ClasificacionBA.predict(X_validation)
    exactitudAlgoritmo = accuracy_score(Y_validation, Y_ClasificacionBA)
    # Matriz de clasificación
    ModeloClasificacion = ClasificacionBA.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),
                                       ModeloClasificacion,
                                       rownames=['Reales'],
                                       colnames=['Clasificación'])
    return Matriz_Clasificacion, exactitudAlgoritmo, ClasificacionBA


def crearGraficaArbol(ClasificacionAD, seleccionCaracteristicas):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    plt.figure(figsize=(16, 16))
    plot_tree(ClasificacionAD, feature_names=seleccionCaracteristicas)
    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    image_data = base64.b64encode(
        bytes_obj.getvalue()).decode('utf-8')
    return image_data


def crearGraficaROC(ClasificacionAD, ClasificacionBA, X_validation, Y_validation):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    # Se crea la gráfica ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(ClasificacionAD,
                                   X_validation,
                                   Y_validation,
                                   ax=ax)
    metrics.RocCurveDisplay.from_estimator(ClasificacionBA,
                                           X_validation,
                                           Y_validation,
                                           ax=ax)
    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    image_data = base64.b64encode(
        bytes_obj.getvalue()).decode('utf-8')
    return image_data
