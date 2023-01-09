# Para la manipulación y análisis de datos
import pandas as pd
from pandas import DataFrame
# Para crear vectores y matrices n dimensionales
import numpy as np
# Para el cálculo de distancias
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


def getMatDist(datosCompletos: DataFrame, tipoDistancia, seleccionCaracteristicas: list):

    # Se estandarizan los datos con las caracteristicas seleccionadas:
    # Se instancia el objeto StandardScaler o MinMaxScaler
    estandarizar = StandardScaler()
    datosSeleccionados = datosCompletos[seleccionCaracteristicas]
    datosEstandarizados = estandarizar.fit_transform(datosSeleccionados)

    if tipoDistancia in ['euclidean', 'chebyshev', 'cityblock']:
        DstEuclidiana = cdist(datosEstandarizados,
                              datosEstandarizados, metric=tipoDistancia)
    elif tipoDistancia == 'minkowski':
        DstEuclidiana = cdist(datosEstandarizados,
                              datosEstandarizados, metric=tipoDistancia, p=1.5)
    else:
        return DataFrame()  # DataFrame vacio para representar error

    return DataFrame(DstEuclidiana)


def getDistObjetos(datos: DataFrame, tipoDistancia, indexObje1, indexObje2):
    Objeto1 = datos.iloc[indexObje1]
    Objeto2 = datos.iloc[indexObje2]
    if tipoDistancia == 'euclidean':
        return distance.euclidean(Objeto1, Objeto2)
    elif tipoDistancia == 'chebyshev':
        return distance.chebyshev(Objeto1, Objeto2)
    elif tipoDistancia == 'cityblock':
        return distance.cityblock(Objeto1, Objeto2)
    elif tipoDistancia == 'minkowski':
        return distance.minkowski(Objeto1, Objeto2, p=1.5)
    else:
        return None  # None para representar error
