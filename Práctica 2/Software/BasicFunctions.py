import numpy as np
from sklearn import neighbors
from sklearn import cross_validation

# Función para cambiar el valor i-ésimo de s
def flip(s, i):
    s[i] = not s[i]

# Obtención de la tasa de acierto para unos datos de entrenamiento con Leave One Out
def getRateL1O(data,labels):
    rate = 0
    leave_1_out_iterators = cross_validation.LeaveOneOut(len(data))
    # Creación del clasificador
    nbrs =  neighbors.KNeighborsClassifier(3)
    # Para cada dato, hacemos kNN
    for train_index, test_index in leave_1_out_iterators:
        d_train = data[train_index]
        d_test = data[test_index]
        l_train = labels[train_index]
        l_test = labels[test_index]
        # Entrenamiento y obtención de la tasa de acierto
        nbrs.fit(d_train, l_train)
        rate += nbrs.score(d_test,[l_test])

    return rate

# Realizar permutación sobre un vector
def permutation(x, random):
    for i in range(len(x)):
        j = random.Randint(0, len(x)-1)
        x[i], x[j] = x[j], x[i]
    return x

# Realizar particiones en un conjunto repartiendo según las etiquetas
def makePartitions(data, categories, random, num_partitions = 2):
    # Calculamos la longitud que tendrá cada partición
    data_copy = np.copy(data)
    len_partition = len(data)//num_partitions

    # Formamos una lista de categorías
    set_cat = list(set(categories))
    num_categories = len(set_cat)

    # Contamos el número de datos de cada clase para enviarlo posteriormente a cada partición
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
        row = permutation(row, random)

    #Repartimos en las particiones
    for i in range(num_partitions-1):
        for j in range(num_categories):
            partitions_d[i].append(data_categorized[j][(i*len_cat_partitions[j]):((i+1)*len_cat_partitions[j])])
            partitions_c[i].append(np.repeat(set_cat[j],len_cat_partitions[j]))

    for j in range(num_categories):
        partitions_d[num_partitions-1].append(data_categorized[j][(num_partitions-1)*len_cat_partitions[j]:])
        partitions_c[num_partitions-1].append(np.repeat(set_cat[j],list(categories).count(set_cat[j])-(num_partitions-1)*len_cat_partitions[j]))

    partitions_d=[np.array([item for sublist in partition for item in sublist ],float) for partition in partitions_d]
    partitions_c=[np.array([item for sublist in partition for item in sublist ]) for partition in partitions_c]

    return([np.array(partitions_d,object),np.array(partitions_c,object)])
