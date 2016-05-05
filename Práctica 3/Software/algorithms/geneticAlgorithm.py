import numpy as np
from knnGPU.knnLooGPU import *
from math import floor
from BasicFunctions import *

def twoPointsCrossOperator(p1, p2):
    num_features = len(p1['chromosome'])
    genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]
    desc = np.zeros(2, dtype=genes_type)

    a, b = sorted(np.random.choice(np.arange(1,num_features),
                                   size=2, replace=False))
    desc[0]['chromosome'] = np.concatenate((p1['chromosome'][:a], p2['chromosome'][a:b],
                             p1['chromosome'][b:]))
    desc[1]['chromosome'] = np.concatenate((p2['chromosome'][:a], p1['chromosome'][a:b],
                             p2['chromosome'][b:]))

    return desc

def huxCrossOperator(p1, p2):
    num_features = len(p1['chromosome'])
    genes_type = [('chromosome',str(num_features)+'bool'),('score', np.float)]
    desc = np.zeros(2, dtype=genes_type)

    for j in range(num_features):
        gen_p1 = p1['chromosome'][j]
        gen_p2 = p2['chromosome'][j]
        if gen_p1 == gen_p2:
            desc[0]['chromosome'][j] = gen_p1
            desc[1]['chromosome'][j] = gen_p1
        else:
            gen = np.random.random() < 0.5
            desc[0]['chromosome'][j] = gen
            desc[1]['chromosome'][j] = not gen

    return desc

def getInitialPopulation(num_genes, num_chromosomes):
    population = np.array([ np.array(np.random.random(size = num_genes) < 0.5)
                           for i in range(num_chromosomes) ])

    return population

def tournament(pair):
    if pair[0]['score'] > pair[1]['score'] or (pair[0]['score'] == pair[1]['score'] and
                    sum(pair[0]['chromosome']) < sum(pair[0]['chromosome'])):
        return pair[0]
    elif pair[1]['score'] > pair[0]['score'] or (pair[0]['score'] == pair[1]['score'] and
                    sum(pair[1]['chromosome']) < sum(pair[0]['chromosome'])):
        return pair[1]
    else:
        return pair[np.random.randint(2)]


def mutate(descendants, mutation_prob):
    num_genes = len(descendants[0]['chromosome'])
    num_descendants = len(descendants)
    num_total_genes = num_descendants*num_genes
    num_genes_to_mutate = floor(num_total_genes*mutation_prob)

    if np.random.random() < (num_total_genes*mutation_prob-num_genes_to_mutate):
        num_genes_to_mutate += 1

    genes_to_mutate = np.random.choice(np.arange(num_total_genes),replace=False,
                                          size = num_genes_to_mutate)

    for gen in genes_to_mutate:
        individual = gen//num_genes
        flip(descendants[individual]['chromosome'],gen%num_genes)
        descendants[individual]['score']=0


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

        # Reordenación
        population.sort(order = 'score')


    return population[-1]['chromosome'], population[-1]['score']
