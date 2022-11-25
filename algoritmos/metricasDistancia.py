# Para la manipulación y análisis de datos
import pandas as pd
from pandas import DataFrame
# Para crear vectores y matrices n dimensionales
import numpy as np
# Para el cálculo de distancias
from scipy.spatial.distance import cdist
from scipy.spatial import distance


def getMatDist(datos: DataFrame, tipoDistancia):
    if tipoDistancia in ['euclidean','chebyshev','cityblock']:
        DstEuclidiana = cdist( datos, datos, metric=tipoDistancia)
    elif tipoDistancia == 'minkowski':
        DstEuclidiana = cdist( datos, datos, metric=tipoDistancia, p=1.5)
    else:
        return DataFrame() # DataFrame vacio para representar error
    
    return DataFrame(DstEuclidiana)