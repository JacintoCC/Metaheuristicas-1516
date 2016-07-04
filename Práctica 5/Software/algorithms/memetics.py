import numpy as np
from knnGPU.knnLooGPU import *
from math import floor
from BasicFunctions import *


def memeticAlgorithm(train_data, train_categ, scorer,
                     selectionOperator, crossOperator,
                     mutationOperator, replaceOperator,
                     num_generations, ls_prob):
    # Parameters
    num_features = len(train_data[0])
    num_chromosomes = 30
    max_checks = 15000
    mutation_prob = 0.001

    genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]

    # Getting initial population
    pop_initial = getInitialPopulation(num_features, num_chromosomes)
    pop_scores = np.array([scorer(train_data[:,chromosome], train_categ)
                                   for )chromosome in pop_initial])

    population = np.array([x for x in zip(pop_initial, pop_scores)], dtype = genes_type)

    num_checks = num_chromosomes
    population.sort(order='score')

    current_generations = 0

    while num_checks<max_checks:
        # Selección
        selected_parents = selectionOperator(population)

        # Cruce
        descendants = crossOperator(selected_parents)

        # Mutación
        mutationOperator(descendants, mutation_prob)

        # Evaluación
        for desc in descendants:
            if desc['score'] == 0 :
                desc['score'] = scorer(train_data[:,desc['chromosome']],train_categ)
                num_checks += 1

        # Reemplazamiento
        population = replaceOperator(population, descendants)

        # Búsqueda local
        if current_generations == num_generations:
            for ind in population:
                if random.random() < ls_prob:
                    ind['chromosome'], ind['score'], ls_checks  = localSearch(train_data, train_categ,
                                                                              scorer, ind['chromosome'])

                    num_checks += ls_checkss

            current_generations = 0
        else:
            current_generations += 1

        # Reordenación
        population.sort(order = 'score')

    return population[-1]['chromosome'], population[-1]['score']


def memetic1(train_data, train_categ, scorer):
    return memeticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational,
                            10, 1)

def memetic01(train_data, train_categ, scorer):
    return memeticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational,
                            10, 0.1)
