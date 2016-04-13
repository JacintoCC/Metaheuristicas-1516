import numpy as np
from sklearn import neighbors
from sklearn import cross_validation
from tempfile import NamedTemporaryFile
import shutil
import csv

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

# Función para actualizar el fichero .csv con las medias
def meansToCSV(name_alg, name_db, mean_results):
    alg_to_index = {'KNN':1,'SFS':2, 'BMB':3,'GRASP':4,'ILS':5}
    db_to_index = {'w':2,'l':6, 'a':10}

    alg_index = alg_to_index[name_alg]
    db_index = db_to_index[name_db]

    filename = 'Resultados/results.csv'
    r = csv.reader(open(filename))
    lines = [l for l in r]

    lines[alg_index][db_index]= str('%.4f' % mean_results[0])
    lines[alg_index][db_index+1]= str('%.4f' % mean_results[1])
    lines[alg_index][db_index+2]= str('%.4f' % mean_results[2])
    lines[alg_index][db_index+3]= str('%.4f' % mean_results[3])
    
    writer = csv.writer(open(filename, 'w'))
    writer.writerows(lines)

# Función para escribir los resultados en un fichero .csv
def  resultsToCSV(name_alg, name_db, results):
    f = open('Resultados/'+name_db+name_alg+'.csv','w')
    f.write("partition,in,out,red,T\n")

    for i in range(len(results)):
        row = 'Particion ' + str(i//2+1) + '-' + str(i%2+1)
        for num in results[i]:
            row += ', ' + str('%.4f' % num)
        f.write(row +  '\n')

    mean_results = np.mean(results, axis=0)
    max_results = results.max(axis=0)
    min_results = results.max(axis=0)
    std_results = np.std(results, axis=0)
    f.write('Media, ' + str('%.4f' % mean_results[0]) + ', ' + str('%.4f' % mean_results[1]) + ', ' + str('%.4f' % mean_results[2]) + ', ' + str('%.4f' % mean_results[3]) + '\n')
    f.write('Max, ' + str('%.4f' % max_results[0]) + ', ' + str('%.4f' % max_results[1]) + ', ' + str('%.4f' % max_results[2]) + ', ' + str('%.4f' % max_results[3]) + '\n')
    f.write('Min, ' + str('%.4f' % min_results[0]) + ', ' + str('%.4f' % min_results[1]) + ', ' + str('%.4f' % min_results[2]) + ', ' + str('%.4f' % min_results[3]) + '\n')
    f.write('Desv. Típica,'  + str('%.4f' % std_results[0]) + ', ' + str('%.4f' % std_results[1]) + ', ' + str('%.4f' % std_results[2]) + ', ' + str('%.4f' % std_results[3]) + '\n')
    f.close()

    meansToCSV(name_alg, name_db, mean_results)
