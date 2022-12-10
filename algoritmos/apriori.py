# Import del algoritmo
from apyori import apriori
import pandas as pd
from pandas import DataFrame

def getChartVariables(TransDF:DataFrame):
    Transacciones = TransDF.values.reshape(-1).tolist()
    Lista = DataFrame(Transacciones)
    Lista['Frecuencia'] = 1

    # Se agrupa los elementos
    Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(
        by=['Frecuencia'], ascending=True)  # Conteo
    Lista['Porcentaje'] = (Lista['Frecuencia'] /
                        Lista['Frecuencia'].sum())  # Porcentaje
    Lista = Lista.rename(columns={0: 'Item'})
    datosX = Lista['Frecuencia']
    Lista.rename(columns={0: 'Item'})
    datosY = Lista['Item']
    return datosX.tolist(), datosY.tolist()

def obtenerApriori(datos:DataFrame, soporteMinimo, confianzaMinima, elevacionMinima):
    # Se crea una lista de listas (que lo requiere el algoritmo apriori)
    listaDatos = datos.stack().groupby(level=0).apply(list).tolist()
    try:
        ReglasAsociacion = apriori(listaDatos,
                                min_support=soporteMinimo,
                                min_confidence=confianzaMinima,
                                min_lift=elevacionMinima)
        data = []
        cols = ["Items", "Soporte", "Confianza", "Elevación"]
        for item in ReglasAsociacion:
            # El primer índice de la lista
            Emparejar = item[0]
            data.append([", ".join(Emparejar), str(
                item[1]), str(item[2][0][2]), str(item[2][0][3])])
            # Antecedente de la regla:
            # print("Antecedente: " + str(item[2][0][0]))
            # print("Concecuente: " + str(item[2][0][1])) 
        datosX, datosY = getChartVariables(datos)
        # Se regresa a la funcion de la API el dataframe con las reglas de asociacion y el numero de reglas generadas:
        return pd.DataFrame(data, columns=cols), len(data), datosX, datosY
    except:
        return