import pandas as pd
from pandas import DataFrame
from seaborn import heatmap
# Para crear vectores y matrices n dimensionales
import numpy as np
# Para generar gráficas a partir de los datos
import matplotlib.pyplot as plt
import io


def obtenerMapaCalor(datos: DataFrame):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    plt.figure(figsize=(14, 7))
    MatCorr = datos.corr()
    mData = np.triu(MatCorr)
    # Se genera el mapa de calor
    heatmap(MatCorr, cmap='RdBu_r', annot=True, mask=mData)
    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    return bytes_obj
