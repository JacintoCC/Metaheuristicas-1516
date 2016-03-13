from math import sqrt
import numpy as np
from random import *

def distance(x,y):
    return sqrt(np.sum((x-y)*(x-y)))

def countingVotes(mins, categories):
    count_votes = dict((i, categories.count(i)) for i in [categories[j] for j in mins])
    most_voted = [ cat for cat in count_votes if count_votes[cat]==max(count_votes.values())]

    for i in mins:
        if( categories[i] in most_voted):
            return categories[i]

def kNN(k, data, categories, item):
    distances =  np.array([[i, distance(data[i],item)] for i in range(len(data))], float)
    mins = np.array([], int)

    for i in range(k):
        distances_i = np.array([distances[j] for j in range(len(data)-1) if not j in mins], float)
        loc_min_distance = np.array([row[1] for row in distances_i], float).argmin()
        mins.concatenate(distances_i[loc_min_distance][0])

    return countingVotes(mins, categories)

def greedySFS(train_data, train_categ):
    num_features = len(training_data[0])

    solution = np.zeros(num_features, int)
    profit_v = np.zeros(num_features, int)
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        for feat in [i for i in range(len(profit_v)) if solution[i]==0]:
            for i in range(len(train_data)):
                train_feats = np.array([[train_data[j,k] for k in range(num_features) if (k==feat or solution[k]==1)] for j in range(len(train_data)) if j != i], float)
                item = np.array([train_data[i,j] for j in range(num_features) if (j==feat or solution[j]==1)], float)
                correct = (kNN(3,train_feats, train_categ, item) == train_categ[i])
                if(correct):
                    profit_v[feat] += 1
        current_profit = profit_v.max()
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[profit_v.argmax()] = 1

    return solution

def runSFS(data, categories, iterations = 10):
    corrects_vector = np.array([], int)

    for i in np.arange(iterations,dtype=int):
        print("Iteration ", i)
        rnd_subject = np.array(sample(np.arange(len(data),dtype=int), len(data)//2), int)
        training_data = np.array([data[j] for j in rnd_subject ], float)

        eval_data = np.array([data[j] for j in range(len(data)) if not j in rnd_subject], float)
        eval_categ = np.array([categories[j] for j in range(len(data)) if not j in rnd_subject], float)

        solution = greedySFS(training_data, training_categories)
        corrects = 0

        for j in range(len(eval_data)):
            train_feats = np.array([[row[j] for j in solution if j==1] for row in training_data], float)
            if(kNN(3,train_feats, train_categ, eval_data[i]) == eval_categ[i]):
                corrects += 1

        np.concatenate(corrects_vector,corrects/len(eval_data)*100)

    return corrects_vector.mean()
