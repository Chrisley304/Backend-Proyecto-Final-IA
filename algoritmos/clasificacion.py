import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score


def getClasificacion(Datos: DataFrame, seleccionCaracteristicas: list, tamanioMuestra, variableClase, filename):
    # Se obtienen los valores de la variable clase
    valoresVariableClase = list(Datos[variableClase].unique())
    # Se verifica que solo existan 2 clases
    if len(valoresVariableClase) > 2:
        return {"error": "La variable predictora tiene mas de 2 clases, el algoritmo solo recibe variables predictorias binarias."}
    else:
        # Se convierten los valores de la variable predictora a 0 y 1
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
                                                                                    random_state=1234,
                                                                                    shuffle=True)
    # Se entrena el modelo
    ClasificacionRL = linear_model.LogisticRegression()
    ClasificacionRL.fit(X_train, Y_train)
    # Predicciones probabilísticas de los datos de prueba
    Probabilidad = ClasificacionRL.predict_proba(X_validation)
    pd.DataFrame(Probabilidad).round(4)
    # Clasificación final
    Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
    # print(Y_ClasificacionRL)
    # Se calcula la exactitud promedio de la validación
    exactitudPromedio = accuracy_score(Y_validation, Y_ClasificacionRL)
    ModeloClasificacion = ClasificacionRL.predict(X_validation)
    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),
                                       ModeloClasificacion,
                                       rownames=['Reales'],
                                       colnames=['Clasificación'])
    # Se crea la gráfica ROC
    datos_grafica = crearGraficaROC(
        ClasificacionRL, X_validation, Y_validation, filename)
    return {"csv": Matriz_Clasificacion.to_csv(), "exactitudPromedio": exactitudPromedio, "graficaROC": datos_grafica}


def crearGraficaROC(ClasificacionRL, X_validation, Y_validation, filename):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    # Se crea la gráfica ROC
    CurvaROC = RocCurveDisplay.from_estimator(
        ClasificacionRL, X_validation, Y_validation, name="Grafica ROC de la clasificacion del archivo " + filename)
    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    image_data = base64.b64encode(
        bytes_obj.getvalue()).decode('utf-8')
    return image_data
