import numpy as np
from knnGPU.knnLooGPU import *
from math import floor
from BasicFunctions import *
from algorithms.generationalGA import *
from algorithms.localSearch import *

def memeticAlgorithm(train_data, train_categ, scorer,
                     selectionOperator, crossOperator,
                     mutationOperator, replaceOperator,
                     localSeachOperator):
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

    current_generation = 0

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

        # Local search
        ls_checks = localSeachOperator(current_generation, population,
                                       train_data, train_categ,scorer)
        num_checks += ls_checks

        # Reordenación
        population.sort(order = 'score')

        current_generation += 1

    return population[-1]['chromosome'], population[-1]['score']


def getLSOperator(num_generations, prob_ls):
    def localSeachOperator(current_generation, population,
                           train_data, train_categ, scorer):
        num_checks = 0
        num_agents_to_ls = round(len(population)*prob_ls)

        # Búsqueda local
        if current_generation % num_generations == 0:
            agents_to_ls = np.random.choice(np.arange(len(population)),replace=False,
                                            size = num_agents_to_ls)
            for i in agents_to_ls:
                population[i]['chromosome'], population[i]['score'], ls_checks  = localSearch(train_data, train_categ,
                                                                          scorer, population[i]['chromosome'])
                num_checks += ls_checks

        return num_checks

    return localSeachOperator

def getLSBestOperator(num_generations, prob_ls):
    def localSeachOperator(current_generation, population,
                           train_data, train_categ, scorer):
        num_checks = 0
        num_agents_to_ls = round(len(population)*prob_ls)

        # Búsqueda local
        if current_generation % num_generations == 0:
            # Reordenación
            population.sort(order = 'score')

            for agent in population[:num_agents_to_ls]:
                agent['chromosome'], agent['score'], ls_checks  = localSearch(train_data, train_categ,
                                                                              scorer, agent['chromosome'])
                num_checks += ls_checks

        return num_checks

    return localSeachOperator

def memetic1(train_data, train_categ, scorer):
    return memeticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational,
                            getLSOperator(10, 1))

def memetic01(train_data, train_categ, scorer):
    return memeticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational,
                            getLSOperator(10, 0.1))

def memetic01mej(train_data, train_categ, scorer):
    return memeticAlgorithm(train_data, train_categ, scorer,
                            selectionOp_Generational,
                            crossOp_Generational(huxCrossOperator),
                            mutate, replaceOp_Generational,
                            getLSBestOperator(10, 0.1))
