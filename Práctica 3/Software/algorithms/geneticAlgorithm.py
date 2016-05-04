import numpy as np
# from knnGPU.knnLooGPU import *
from math import floor
from BasicFunctions import *

def twoPointsCrossOperator(parents):
    num_features = len(parents[0]['chromosome'])
    num_descendants = len(parents)
    descendants = np.zeros((num_descendants,num_features), dtype=np.bool)

    for i in range(int(num_descendants//2)):
        parent1 = parents[2*i]
        parent2 = parents[2*i+1]

        a, b = sorted(np.random.choice(np.arange(1,num_features),
                                       size=2, replace=False))

        descendants[2*i]   = np.concatenate((parent1['chromosome'][:a],
                                             parent2['chromosome'][a:b],
                                             parent1['chromosome'][b:]))
        descendants[2*i+1] = np.concatenate((parent2['chromosome'][:a],
                                             parent1['chromosome'][a:b],
                                             parent2['chromosome'][b:]))

    return descendants

def huxCrossOperator(parents):
    num_features = len(parents[0]['chromosome'])
    num_descendants = len(parents)
    descendants = np.zeros((num_descendants,num_features), dtype=np.bool)

    for i in range(int(num_descendants//2)):
        parent1 = parents[2*i]
        parent2 = parents[2*i+1]

        for j in range(num_features):
            gen_p1 = parent1['chromosome'][j]
            gen_p2 = parent2['chromosome'][j]
            if gen_p1 == gen_p2:
                descendants[2*i][j]   = gen_p1
                descendants[2*i+1][j] = gen_p1
            else:
                gen = np.random.random() < 0.5
                descendants[2*i][j]   = gen
                descendants[2*i+1][j] = not gen

    return descendants

def getInitialPopulation(num_genes, num_chromosomes):
    population = np.array([ np.array(np.random.random(size = num_genes) < 0.5)
                           for i in range(num_chromosomes) ])

    return population

def tournament(pair):
    if pair[0]['score'] > pair[1]['score']:
        return pair[0]
    elif pair[0]['score'] < pair[1]['score']:
        return pair[1]
    else:
        if sum(pair[0]['chromosome']) < sum(pair[0]['chromosome']):
            return pair[0]
        elif sum(pair[0]['chromosome']) < sum(pair[0]['chromosome']):
            return pair[1]
        else:
            return pair[np.random.randint(2)]


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

    genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]

    # Getting initial population
    pop_initial = getInitialPopulation(num_features, num_chromosomes)
    pop_scores = np.array([scorer(train_data[:,chromosome], train_categ)
                                  for chromosome in pop_initial])

    population = np.array([x for x in zip(pop_initial, pop_scores)], dtype = genes_type)

    num_checks = num_chromosomes
    population.sort(order='score')

    while num_checks<max_checks:
        # Selección
        selected_parents = selectionOperator(population)

        # Cruce
        desc_chrom = crossOperator(selected_parents)

        # Mutación
        mutationOperator(desc_chrom, mutation_prob)

        # Evaluación
        desc_scores = [scorer(train_data[:,desc], train_categ)
                       for desc in desc_chrom]
        num_checks += len(desc_chrom)

        descendants = np.array([x for x in zip(desc_chrom, desc_scores)], dtype = genes_type)

        # Reemplazamiento
        replaceOperator(population, descendants)

        # Reordenación
        population.sort(order = 'score')


    return population[-1]['chromosome'], population[-1]['score']
