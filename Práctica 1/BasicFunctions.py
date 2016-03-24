import numpy as np
"""
    Importación del generador de números aleatorios en C
"""
try:
    from ctypes import *
except ImportError:
    print('ERROR! La biblioteca *ctypes* para Python no esta disponible.')
    sys.exit(-1)

random_ppio = cdll.LoadLibrary('./Random_ppio/random_ppio.so')
random_ppio.Set_random(124)

def permutation(x):
    for i in range(len(x)):
        j = random_ppio.Randint(0, len(x)-1)
        x[i], x[j] = x[j], x[i]
    return x

def makePartitions(data, categories, num_partitions = 2):
    # Calculamos la longitud que tendrá cada partición
    data_copy = np.copy(data)
    len_partition = len(data)//num_partitions

    # Formamos una lista de categorías
    set_cat = list(set(categories))
    num_categories = len(set_cat)
    dict_categories = dict([(set_cat[i],i) for i in range(num_categories)])
    len_cat_partitions = [list(categories).count(i)//num_partitions for i in set_cat]
    data_categorized = [[] for i in range(num_categories)]

    # Creación de las estructuras que devolveremos
    partitions_d = [[] for i in range(num_partitions)]
    partitions_c = [[] for i in range(num_partitions)]

    # Categorizamos los elementos
    for i in range(len(categories)):
        num_cat = dict_categories[categories[i]]
        data_categorized[num_cat].append(data_copy[i])

    # Hacemos una permutación por los datos categorizados
    for row in data_categorized:
        row = permutation(row)

    #Repartimos en las particiones
    for i in range(num_partitions-1):
        for j in range(num_categories):
            partitions_d[i].append(data_categorized[j][(i*len_cat_partitions[j]):((i+1)*len_cat_partitions[j])])
            partitions_c[i].append(np.repeat(set_cat[j],len_cat_partitions[j]))

    for j in range(num_categories):
        partitions_d[num_partitions-1].append(data_categorized[j][(num_partitions-1)*len_cat_partitions[j]:])
        partitions_c[num_partitions-1].append(np.repeat(set_cat[j],list(categories).count(set_cat[j])-(num_partitions-1)*len_cat_partitions[j]))

    partitions_d=[[item for sublist in partition for item in sublist ] for partition in partitions_d]

    return([np.array(partitions_d,object),np.array(partitions_c,object)])
