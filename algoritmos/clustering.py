import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import io
import base64


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
            graficaClusters = obtenerGrafica(datosEstandarizados, MJerarquico)
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
        graficaClusters = obtenerGrafica(datosEstandarizados, MParticional)
        # graficaClusters = obtenerGraficaParticional(
        #     MParticional, datosEstandarizados)
    else:
        return {"error": "El tipo de clustering {} es incorrecto (Solo puedes elegir jerarquico o particional)".format(tipoClustering)}

    # Cantidad de elementos en los clusters
    clustersConteo = (datos.groupby(['cluster'])['cluster'].count()).to_dict()
    # Se obtiene la gráfica de los elementos y los centros de los clusters
    return {"csv": datos.to_csv(), "conteo": clustersConteo, "grafica": graficaClusters}


def obtenerGrafica(MEstandarizada, MModelo):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    # Se comienza a crear la grafica
    plt.figure(figsize=(10, 7))
    plt.scatter(MEstandarizada[:, 0],
                MEstandarizada[:, 1], c=MModelo.labels_)
    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    # Se convierte a base64
    image_data = base64.b64encode(
        bytes_obj.getvalue()).decode('utf-8')
    return image_data


# Gráfica de los elementos y los centros de los clusters
def obtenerGraficaParticional(MParticional, MEstandarizada):
    # Para evitar errores en el servidor
    plt.switch_backend('Agg')
    # Se comienza a crear la grafica
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.style.use('ggplot')
    coloresAll = ['red', 'blue', 'green', 'yellow', 'orange',
                  'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                  'red', 'blue', 'green', 'yellow', 'orange',
                  'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    maxCluster = max(MParticional.labels_)
    colores = coloresAll[:maxCluster+1]
    asignar = []
    for row in MParticional.labels_:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(MEstandarizada[:, 0],
               MEstandarizada[:, 1],
               MEstandarizada[:, 2], marker='o', c=asignar, s=60)
    ax.scatter(MParticional.cluster_centers_[:, 0],
               MParticional.cluster_centers_[:, 1],
               MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)

    # Se guarda la imagen en Bytes
    bytes_obj = io.BytesIO()
    plt.savefig(bytes_obj, format='png')
    bytes_obj.seek(0)
    image_data = base64.b64encode(
        bytes_obj.getvalue()).decode('utf-8')
    # columnas = datos.columns.to_list()
    return image_data
