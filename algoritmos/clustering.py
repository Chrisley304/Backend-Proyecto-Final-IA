import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator


def getClustering(datos: DataFrame, tipoClustering, minClusters, maxClusters, algoritmoDistancia, seleccionCaracteristicas):
    MatrizDatos = numpy.array(datos)
    estand = StandardScaler()
    # Se crean las etiquetas de los elementos en los clústeres
    datosEstandarizados = estand.fit_transform(MatrizDatos)
    if tipoClustering == 'jerarquico':
        if algoritmoDistancia in ['euclidean', 'chebyshev', 'cityblock']:
            MJerarquico = AgglomerativeClustering(
                n_clusters=maxClusters, linkage='complete', affinity=algoritmoDistancia)
            MJerarquico.fit_predict(datosEstandarizados)
            datos['cluster'] = MJerarquico.labels_
        else:
            return {"error": "El algoritmo de distancia {} no es valido (Solo puedes utilizar 'euclidean','chebyshev','cityblock')".format(algoritmoDistancia)}
    elif tipoClustering == 'particional':
        SSE = []
        # Se utiliza random_state para inicializar el generador interno de números aleatorios
        for i in range(minClusters, maxClusters):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(datosEstandarizados)
            SSE.append(km.inertia_)
        # Se localiza la rodilla:
        kl = KneeLocator(range(minClusters, maxClusters), SSE,
                         curve="convex", direction="decreasing")
        MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(
            datosEstandarizados)
        MParticional.predict(datosEstandarizados)
        datos['cluster'] = MParticional.labels_
    else:
        return {"error": "El tipo de clustering {} es incorrecto (Solo puedes elegir jerarquico o particional)".format(tipoClustering)}

    # Cantidad de elementos en los clusters
    clustersConteo = (datos.groupby(['cluster'])['cluster'].count()).to_dict()
    return {"csv": datos.to_csv(), "conteo": clustersConteo}
