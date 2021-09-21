import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd

from AckleyFunction import ackley_fun
from UP_VH import UnPadreVariosHijos
from VP_VH import VariosPadresVariosHijos
from VP_VH_T import VariosPadresVariosHijosTraslape


def seleccionar_modelo():
    if len(sys.argv) < 2:
        return None
    modeloIdx = int(sys.argv[1])
    if modeloIdx == 1:
        return UnPadreVariosHijos
    if modeloIdx == 2:
        return VariosPadresVariosHijos
    if modeloIdx == 3:
        return VariosPadresVariosHijosTraslape
    return None


modelo = seleccionar_modelo()
if modelo is None:
    print(
        "Introduzca un argumento {1, 2, 3} para seleccionar un algoritmo de mutación")
    exit(1)
dfExperimentos = pd.DataFrame()
experimentos = 10

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
logger = logging.getLogger()

for experimento in range(1, experimentos + 1):
    algoritmo = modelo(ackley_fun)
    mejores, evaluaciones = algoritmo.run()

    cantidad = len(mejores)

    logger.info('Experimento %d', experimento)
    logger.info('Cantidad de soluciones %d', cantidad)
    logger.info('Cantidad de evaluaciones %d', len(evaluaciones))

    df = pd.DataFrame()
    df['algoritmo'] = [algoritmo.__class__.__name__] * cantidad
    df['experimento'] = [experimento] * cantidad
    df['iteracion'] = list(range(cantidad))
    df['x'] = mejores
    df['evaluacion'] = evaluaciones
    df.at[0, 'menor'] = df.loc[0]['evaluacion']
    for rowidx in range(1, df.shape[0]):
        df.at[rowidx, 'menor'] = min(
            df.loc[rowidx]['evaluacion'], df.iloc[rowidx - 1].menor)
    dfExperimentos = dfExperimentos.append(df)

dfExperimentos.reset_index(drop=True, inplace=True)

resultados = dfExperimentos.groupby(
    'iteracion').agg({'menor': ['mean', 'std']})
promedios = resultados['menor']['mean'].values
std = resultados['menor']['std'].values
plt.plot(range(cantidad), promedios, color='red', marker='*')
plt.plot(range(cantidad), promedios + std, color='b', linestyle='-.')
plt.plot(range(cantidad), promedios - std, color='b', linestyle='-.')
plt.xlabel('iteraciones')
plt.ylabel('menor encontrado')
plt.legend(['promedio', 'promedio + std', 'promedio - std'])

if modelo is UnPadreVariosHijos:
    plt.title('Un padre, varios hijos')
    plt.savefig('upvh.png')
elif modelo is VariosPadresVariosHijos:
    plt.title('Varios padres, varios hijos')
    plt.savefig('vpvh.png')
elif modelo is VariosPadresVariosHijosTraslape:
    plt.title('Varios padres, varios hijos, con traslape')
    plt.savefig('vpvht.png')
