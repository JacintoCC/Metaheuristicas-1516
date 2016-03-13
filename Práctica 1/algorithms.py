from math import sqrt
import numpy as np
from random import *

def distance(x,y):
    dist = 0
    for i in range(len(x)):
        dist = dist + (x[i]-y[i])*(x[i]-y[i])
    return sqrt(dist)

def countingVotes(mins, categories):
    count_votes = dict((i, categories.count(i)) for i in [categories[j] for j in mins])
    most_voted = [ cat for cat in count_votes if count_votes[cat]==max(count_votes.values())]

    for i in mins:
        if( categories[i] in most_voted):
            return categories[i]

def kNN(k, data, categories, item):
    # ¡Por aquí! 3
    distances =  [[i, distance(data[i],item)] for i in range(len(data))]
    mins = []

    for i in range(k):
        distances_i = [distances[j] for j in range(len(data)-1) if not j in mins]
        loc_min_distance = np.array([row[1] for row in distances_i]).argmin()
        mins.append(distances_i[loc_min_distance][0])

    return countingVotes(mins, categories)

def greedySFS(training_data):
    train_categ = [row[0] for row in training_data]

    num_features = len(training_data[0])-1
    train_subj =  [[row[i] for i in range(1,num_features+1)] for row in training_data]

    solution = [0 for i in range(num_features)]
    profit_v = [0 for i in range(num_features)]
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        for feat in [i for i in range(len(profit_v)) if solution[i]==0]:
            for i in range(len(train_subj)):
                train_feats = [[train_subj[j][k] for k in range(num_features) if (k==feat or solution[k]==1)] for j in range(len(train_subj)) if j != i]
                item = [train_subj[i][j] for j in range(num_features) if (j==feat or solution[j]==1)]
                # ¡Por aquí! 2
                correct = (kNN(3,train_feats, train_categ, item) == train_categ[i])
                if(correct):
                    profit_v[feat] += 1
        current_profit = max(profit_v)
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[profit_v.index(current_profit)] = 1

    return solution

def runSFS(data, iterations = 10):
    corrects_vector = []

    for i in range(iterations):
        rnd_subject = sample(range(len(data)),len(data)//2)
        training_data = [data[j] for j in rnd_subject ]

        eval_data = [[data[j][k] for k in range(1,len(data[0]))] for j in range(len(data)) if not j in rnd_subject]
        eval_categ = [data[j][0] for j in range(len(data)) if not j in rnd_subject]

        # ¡Por aquí 1!
        solution = greedySFS(training_data)
        train_categ = [row[0] for row in training_data]
        corrects = 0

        for j in range(len(eval_data)):
            train_feats = [[row[j] for j in solution if j==1] for row in training_data]
            if(kNN(3,train_feats, train_categ, eval_data[i]) == eval_categ[i]):
                corrects += 1

        corrects_vector.append(corrects)

    mean = 0
    for n in corrects:
        mean += n

    return mean/(iterations*len(eval_data)*1.0)*100.0
