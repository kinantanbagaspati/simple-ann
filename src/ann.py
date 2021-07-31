import math
import random

def sigmoid(x, is_prime):
    if(is_prime):
        return x*(1-x)
    if(x<-100):
        return 1/(1 + math.exp(100))
    if(x>100):
        return 1/(1 + math.exp(-100))
    return 1/(1 + math.exp(-x))

def relu(x, is_prime):
    if not(is_prime):
        return max(0, x)
        #return x
    return 1

def MSE(array, target):
    toReturn = 0
    n = len(array)
    for i in range(n):
        toReturn += (array[i] - target[i])**2
    toReturn /= n
    return toReturn

#neurons_per_layer tidak include input dan output
def ann(data, label_col, neurons_per_layer, learning_rate, episodes, is_relu, test):
    #assuming label_col merupakan salah satu kolom
    columns = data.columns
    neurons_per_layer.insert(0, len(columns)-1)
    target_values = data[label_col].unique()
    target = [0 for i in range(len(target_values))]
    neurons_per_layer.append(len(target_values))
    #jumlah neuron per layer sudah ditambahkan
    #print(neurons_per_layer)

    #initializing neurons, weight, bias
    neurons = [[0 for j in range(neurons_per_layer[i])] for i in range(len(neurons_per_layer))]
    #Karena tidak dispesifikasikan, seeding awal dianggap uniform (bukan distribusi normal)
    #weight[i][j][k] menandakan weight antara neuron ke j pada layer i ke neuron ke k pada layer i+1
    weight = [[[random.uniform(-1,1) for k in range(neurons_per_layer[i+1])]
                for j in range(neurons_per_layer[i])]
                for i in range(len(neurons_per_layer)-1)]
    #print(weight)
    bias = [[random.uniform(-1,1) for j in range(neurons_per_layer[i])] for i in range(1, len(neurons_per_layer))]

    
    
    #training time
    for i in range(episodes):
        print("Starting episode " + str(i+1), end = "->")

        tot_loss = 0
        d_weight = [[[0 for k in range(neurons_per_layer[i+1])]
                    for j in range(neurons_per_layer[i])]
                    for i in range(len(neurons_per_layer)-1)]
        d_bias = [[0 for j in range(neurons_per_layer[i])] for i in range(1, len(neurons_per_layer))]
        d_weight_tot = [[[0 for k in range(neurons_per_layer[i+1])]
                    for j in range(neurons_per_layer[i])]
                    for i in range(len(neurons_per_layer)-1)]
        d_bias_tot = [[0 for j in range(neurons_per_layer[i])] for i in range(1, len(neurons_per_layer))]

        for j in range(len(data)):
            #forward propagation
            after_target = False
            for k in range(len(columns)):
                if label_col == columns[k]:
                    after_target = True
                elif after_target:
                    if is_relu:
                        neurons[0][k-1] = relu(data[columns[k]][j], False)
                    else:
                        neurons[0][k-1] = sigmoid(data[columns[k]][j], False)
                else:
                    if is_relu:
                        neurons[0][k] = relu(data[columns[k]][j], False)
                    else:
                        neurons[0][k] = sigmoid(data[columns[k]][j], False)
            
            for k in range(1, len(neurons_per_layer)):
                for l in range(neurons_per_layer[k]):
                    neurons[k][l] = bias[k-1][l]
                    for m in range(neurons_per_layer[k-1]):
                        neurons[k][l] += neurons[k-1][m] * weight[k-1][m][l]
                    if is_relu:
                        neurons[k][l] = relu(neurons[k][l], False)
                    else:
                        neurons[k][l] = sigmoid(neurons[k][l], False)
            #if j == 0:
            #    print(neurons)
            #Calculate loss
            for k in range(len(target)):
                if target_values[k] == data[label_col][j]:
                    target[k] = 1
                else:
                    target[k] = 0
            
            loss = MSE(neurons[len(neurons)-1], target)
            tot_loss += loss

            #back propagation
            for k in range(len(bias)-1, -1, -1):
                #cari sgd bias
                for l in range(neurons_per_layer[k+1]):
                    if(k == len(bias)-1):
                        if is_relu:
                            d_bias[k][l] = 2*(neurons[k+1][l]-target[l])*relu(neurons[k+1][l], True)/len(target)
                        else:
                            d_bias[k][l] = 2*(neurons[k+1][l]-target[l])*sigmoid(neurons[k+1][l], True)/len(target)
                    else:
                        d_bias[k][l] = 0
                        for m in range(neurons_per_layer[k+2]):
                            if is_relu:
                                d_bias[k][l] += relu(neurons[k+1][l], True) * d_weight[k+1][l][m]
                            else:
                                d_bias[k][l] += sigmoid(neurons[k+1][l], True) * d_weight[k+1][l][m]
                    d_bias_tot[k][l] += d_bias[k][l]
                #cari sgd weight
                for l in range(neurons_per_layer[k]):
                    for m in range(neurons_per_layer[k+1]):
                        d_weight[k][l][m] = neurons[k][l]*d_bias[k][m]
                        d_weight_tot[k][l][m] += d_weight[k][l][m]
        
            #updating weights and bias
            for k in range(len(bias)):
                for l in range(neurons_per_layer[k+1]):
                    bias[k][l] -= learning_rate * d_bias[k][l] * loss
                for l in range(neurons_per_layer[k]):
                    for m in range(neurons_per_layer[k+1]):
                        weight[k][l][m] -= learning_rate * d_weight[k][l][m] * loss
        
        print("total loss:", str(tot_loss))
    
    #print(weight)
    
    #Tinggal predict
    result = []
    d_columns = {}
    after_target = False
    for i in range(len(columns)):
        if label_col == columns[i]:
            after_target = True
        elif after_target:
            d_columns[columns[i]] = i
        else:
            d_columns[columns[i]] = i-1

    for i in range(len(test)):
        #forward propagation
        for j in range(len(test.columns)):
            column_id = d_columns[test.columns[j]]
            neurons[0][column_id] = test[test.columns[j]][i]
        #print(neurons[0])
        for k in range(1, len(neurons_per_layer)):
            for l in range(neurons_per_layer[k]):
                neurons[k][l] = bias[k-1][l]
                for m in range(neurons_per_layer[k-1]):
                    neurons[k][l] += neurons[k-1][m] * weight[k-1][m][l]
                if is_relu:
                    neurons[k][l] = relu(neurons[k][l], False)
                else:
                    neurons[k][l] = sigmoid(neurons[k][l], False)
        max_val = -1
        result_id = -1
        for j in range(len(target)):
            print(target_values[j], end = ": ")
            print(neurons[len(neurons)-1][j], end = " ")
            if max_val < neurons[len(neurons)-1][j]:
                max_val = neurons[len(neurons)-1][j]
                result_id = j
        print("Line " + str(i+1) + " labelled " + str(target_values[result_id]))
        result.append(target_values[result_id])
    test[label_col] = result
    
    
        
                