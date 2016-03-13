from math import sqrt
import numpy as np

def distance(x,y):
    dist = 0
    for i in range(0,len(x)):
        dist = dist + (x[i]-y[i])*(x[i]-y[i])
    return sqrt(dist)

def countingVotes(mins, categories):
    count_votes = dict((i, categories.count(i)) for i in [categories[j] for j in mins])
    most_voted = [ cat for cat in count_votes if count_votes[cat]==max(count_votes.values())]

    for i in mins:
        if( categories[i] in most_voted):
            return categories[i]

def kNN(k, data, categories, item):
    distances =  [[i, distance(data[i],item)] for i in range(len(data))]
    mins = []

    for i in range(k):
        distances_i = [distances[j] for j in range(len(data)-1) if not j in mins]
        loc_min_distance = np.array([row[1] for row in distances_i]).argmin()
        mins.append(distances_i[loc_min_distance][0])

    return countingVotes(mins, categories)

def greedySFS(training_data, evaluating_data):
    train_categ = [row[0] for row in training_data]
    eval_categ = [row[0] for row in evaluating_data]

    num_features = len(training_data[0])-1
    train_subj =  [row[1:(num_features+1)] for row in training_data]
    eval_subj =  [row[1:(num_features+1)] for row in evaluating_data]

    solution = [0 for i in range(num_features)]
    profit_v = [0 for i in range(num_features)]
    exists_profit = True
    previous_profit = 0

    while(exists_profit):
        for feat in range(len(profit_v)) if solution[feat]==0:
            train_feats_i = [[row[j] for j in range(num_features) if (j==feat or solution[j]==1)] for row in training_data]
            for i in range(len(eval_subj))
                item = [eval_subj[i][j] for j in range(num_features) if (j==feat or solution[j]==1)]
                correct = (kNN(3,train_feats_i, train_categ, item) == eval_categ[i])
                if(correct):
                    profit_v[feat] += 1
        current_profit = max(profit_v)
        exists_profit = current_profit > previous_profit
        if(exists_profit):
            solution[profit_v.index(current_profit)] = 1

    return solution

def runSFS(data, iterations = 10):
    for i in range(iterations):
        
