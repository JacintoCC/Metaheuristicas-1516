import numpy as np
from knnGPU.knnLooGPU import *
from math import floor
from BasicFunctions import *

def twoPointsCrossOperator(parents):
    num_features = len(parents[0])
    num_descendants = len(parents)
    descendants = np.zeros((num_descendants,num_features), dtype=np.bool)

    for i in range(int(num_descendants//2)):
        parent1 = parents[2*i]
        parent2 = parents[2*i+1]

        a, b = sorted(np.random.choice(np.arange(1,num_features),
                                       size=2, replace=False))

        descendants[2*i]   = np.concatenate((parent1[:a],parent2[a:b],parent1[b:]))
        descendants[2*i+1] = np.concatenate((parent2[:a],parent1[a:b],parent2[b:]))

    return descendants

def huxCrossOperator(parents):
    num_features = len(parents[0])
    num_descendants = len(parents)
    descendants = np.zeros((num_descendants,num_features), dtype=np.bool)

    for i in range(int(num_descendants//2)):
        parent1 = parents[2*i]
        parent2 = parents[2*i+1]

        for j in range(num_features):
            gen_p1 = parent1[j]
            gen_p2 = parent2[j]
            if gen_p1 == gen_p2:
                descendants[2*i][j]   = gen_p1
                descendants[2*i+1][j] = gen_p1
            else:
                gen = np.random.random() < 0.5
                descendants[2*i][j]   = gen
                descendants[2*i+1][j] = gen

    return descendants

def getInitialPopulation(num_genes, num_chromosomes):
    population = np.array([ np.random.random(size = num_genes) < 0.5
                           for i in range(num_chromosomes) ])

    return population

def tournament(pair, scores):
    if scores[pair[0]] > scores[pair[1]]:
        return pair[0]
    elif scores[pair[1]] > scores[pair[0]]:
        return pair[1]
    else:
        return pair[np.random.randint(2)]


def sort_population(population, pop_scores):
    population, pop_scores = (list(x) for x in zip(*sorted(zip(population, pop_scores),
                                                           reverse = True,
                                                           key=lambda pair: pair[1])))

def mutate(descendants, mutation_prob):
    num_genes = len(descendants[0])
    num_descendants = len(descendants)
    num_total_genes = num_descendants*num_genes
    num_genes_to_mutate = floor(num_total_genes*mutation_prob)

    if np.random.random() < (num_total_genes*mutation_prob-num_genes_to_mutate):
        num_genes_to_mutate += 1

    genes_to_mutate = np.random.choice(np.arange(num_total_genes),replace=False,
                                          size = num_genes_to_mutate)

    for gen in genes_to_mutate:
        flip(descendants[gen//num_genes],gen%num_genes)


def geneticAlgorithm(train_data, train_categ, scorer,
                     selectionOperator, crossOperator,
                     mutationOperator, replaceOperator):
    # Parameters
    num_features = len(train_data[0])
    num_chromosomes = 30
    max_checks = 15000
    mutation_prob = 0.001

    # Getting initial population
    population = getInitialPopulation(num_features, num_chromosomes)
    pop_scores = np.array([scorer(train_data[:,chromosome], train_categ)
                                  for chromosome in population])

    num_checks = num_chromosomes
    sort_population(population, pop_scores)

    while num_checks<max_checks:
        # Selección
        selected_parents = selectionOperator(population, pop_scores)

        # Cruce
        descendants = crossOperator(selected_parents)

        # Mutación
        mutationOperator(descendants, mutation_prob)

        # Evaluación
        desc_scores = [scorer(train_data[:,desc], train_categ)
                              for desc in descendants]
        num_checks += len(descendants)

        # Reemplazamiento
        replaceOperator(population, pop_scores, descendants, desc_scores)

        # Reordenación
        sort_population(population, pop_scores)


    return [population[0], pop_scores[0]]
