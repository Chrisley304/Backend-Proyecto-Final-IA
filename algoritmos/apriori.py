# Import del algoritmo
from apyori import apriori
import pandas as pd
from pandas import DataFrame

def obtenerApriori(datos:DataFrame, soporteMinimo, confianzaMinima, elevacionMinima):
    # Se crea una lista de listas (que lo requiere el algoritmo apriori)
    listaDatos = datos.stack().groupby(level=0).apply(list).tolist()
    try:
        ReglasAsociacion = apriori(listaDatos,
                                min_support=soporteMinimo,
                                min_confidence=confianzaMinima,
                                min_lift=elevacionMinima)
        # Se regresa a la funcion de la API la lista con las reglas de asociacion
        return list(ReglasAsociacion)
    except:
        return
    return