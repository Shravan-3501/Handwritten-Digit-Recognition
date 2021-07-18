import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt

#Constants
INPUT_DATA_SIZE = 5000
TRAIN_DATA_SIZE = 4500
TEST_DATA_SIZE = INPUT_DATA_SIZE - TRAIN_DATA_SIZE
NO_OF_FEATURES = int(784)
OUTPUT_LAYER_SIZE = int(10)
NO_OF_LAYERS = 3
NO_OF_FOLDS = 1

CV_DATA_SIZE = int(TRAIN_DATA_SIZE/NO_OF_FOLDS)
HIDDEN_LAYER_NODES_SET = [25]

#Input and output file names
INPUT_FILE_X = "Q2DataX.txt"
INPUT_FILE_Y = "Q2DataY.txt" 
OUTPUT_FILE =  "Q2Output_tempor7.txt"

#Hyperparameters

ITERATIONS_SET = [4000]
LEARNING_RATE_SET = [[1,0.1]]
REGULARISATION_PARAMETER_SET = [[0.01,2]]
#values = [0.01,0.1,1,2,10]
#REGULARISATION_PARAMETER_SET = [[values[i],values[j]] for i in range(5) for j in range(5)]

Train_predictions = True
Test_predictions = True
Output = True

def make_predictions(X,W,Y):
    n = len(X)
    e = np.ones((n,1))
    activation = X
    for l in range(NO_OF_LAYERS-1):
        activation = np.hstack([e,activation])
        activation = np.dot(activation,W[l])
    Y_pred = np.zeros((n,1))
    for iter in range(n):
        temp = 0
        m = activation[iter][0]
        for j in range(OUTPUT_LAYER_SIZE):
            if(activation[iter][j] > m):
                m = activation[iter][j]
                temp = j 
        Y_pred[iter] = temp
    return Y_pred
    
def calc_accuracy(X,Y_act,W,predictions):
    Y_pred = make_predictions(X,W,Y_act)
    n = Y_act.size
    E = np.abs(np.sign(Y_pred-Y_act))
    e = (1/n)*np.sum(E)
    return (1-e)*100

def calc_sig(z):
    y = 1/(1+np.exp(-1*z))
    return y

def initialize_weights():
    W = []
    for l in range(NO_OF_LAYERS - 1):
        epsilon = math.sqrt(6/(s[l]+s[l+1]))
        W_temp = np.random.uniform(-1*epsilon,epsilon,(s[l]+1,s[l+1]))
        W.append(W_temp)
    return W

def visualise(X,Y,Y_pred):
    fig = plt.figure()
    for i in range(50):
        image_transp = X[i].reshape(28,28).transpose()
        fig.add_subplot(10,5,i+1)
        plt.imshow(image_transp,cmap = 'gray')
        plt.title(str("Pred :") + str(Y_pred[i]),fontsize=12)
        plt.axis("off")
    plt.subplots_adjust(hspace = 1.2)
    plt.show()


def neural_network(X_train,Y_train,iterations,learning_rate,reg_para,W):
    e = np.ones((CV_DATA_SIZE,1))
    for i in range(iterations):
        activations = []
        activations.append(np.hstack([e,X_train]))
        for l in range(NO_OF_LAYERS-1):
            temp_activation = calc_sig(np.dot(activations[l],W[l]))
            if(l < NO_OF_LAYERS - 2):
                temp_activation = np.hstack([e,temp_activation])
            activations.append(temp_activation)
        delta = []
        delta_temp = activations[NO_OF_LAYERS-1]
        for j in range(CV_DATA_SIZE):
            x = int(Y_train[j])
            delta_temp[j][x] = delta_temp[j][x] - 1
        delta.append(delta_temp)
        for l in range(NO_OF_LAYERS-2):
            idx = NO_OF_LAYERS - l - 2
            delta_temp = np.dot(delta[l],np.transpose(W[idx]))
            sig_der = np.multiply(activations[idx],1-activations[idx])
            delta_temp = np.multiply(delta_temp,sig_der)
            delta.append(delta_temp[:,1:])
        for l in range(NO_OF_LAYERS - 1):
            idx = NO_OF_LAYERS - l - 2
            Del = np.dot(np.transpose(activations[idx]),delta[l])
            Del = Del + reg_para[idx]*W[idx]
            Del = Del/TRAIN_DATA_SIZE
            W[idx] = W[idx] - learning_rate[idx]*Del
        '''if(i % 100 == 0 and i > 0):
            train_accuracy = calc_accuracy(X_train,Y_train,W,Train_predictions)
            test_accuracy = calc_accuracy(X_test,Y_test,W,Test_predictions)
            create_outputfile(i,learning_rate,reg_para,train_accuracy,test_accuracy)'''
    return W  

def create_outputfile(iterations, learning_rate, reg_para, train_accuracy, test_accuracy):
    outputFile = open(OUTPUT_FILE, 'a')
    z = str(iterations) + '\t' + str(learning_rate) + '\t' + str(reg_para) + '\t' + str(HIDDEN_LAYER_NODES) + '\t' + str(train_accuracy) + '\t' + str(test_accuracy) + '\n' 
    outputFile.write(z)
    outputFile.close()
 
        
            
#Reading data from the input file and storing it
fileData = open(INPUT_FILE_X , "r")
X = np.zeros((INPUT_DATA_SIZE,NO_OF_FEATURES+1))
i = 0
for line in fileData:
    temp = line.split()
    for j in range(0,NO_OF_FEATURES):
        X[i][j] = float(temp[j])
    i = i + 1
fileData = open(INPUT_FILE_Y,"r")
i = 0
for line in fileData:
    temp = line.split()
    X[i][NO_OF_FEATURES] = float(temp[0])
    i = i + 1

np.random.shuffle(X)

X_CV = []
Y_CV = []
X_test = np.zeros((TEST_DATA_SIZE,NO_OF_FEATURES))
Y_test = np.zeros((TEST_DATA_SIZE,1))

for k in range(NO_OF_FOLDS):
    X_temp = np.zeros((CV_DATA_SIZE,NO_OF_FEATURES))
    Y_temp = np.zeros((CV_DATA_SIZE,1))
    for i in range(CV_DATA_SIZE):
        for j in range(0,NO_OF_FEATURES):
            X_temp[i][j] = X[i + (k*CV_DATA_SIZE)][j]
        Y_temp[i] = X[i + (k*CV_DATA_SIZE)][NO_OF_FEATURES]
    X_CV.append(X_temp)
    Y_CV.append(Y_temp)

for i in range(0,TEST_DATA_SIZE):
    for j in range(0,NO_OF_FEATURES):
        X_test[i][j] = X[i+TRAIN_DATA_SIZE][j]
    Y_test[i] = X[i+TRAIN_DATA_SIZE][NO_OF_FEATURES]


#Driver Code   
f = open(OUTPUT_FILE,'w')
f.close()

CV_accuracy = []
Test_accuracy = []               
for learning_rate in LEARNING_RATE_SET:
    for reg_para in REGULARISATION_PARAMETER_SET:
        for iterations in ITERATIONS_SET:
            for HIDDEN_LAYER_NODES in HIDDEN_LAYER_NODES_SET:
                s = []
                s.append(NO_OF_FEATURES)
                for i in range(2,NO_OF_LAYERS):
                    s.append(HIDDEN_LAYER_NODES)
                s.append(OUTPUT_LAYER_SIZE)
                temp_acc = 0
                for i in range(NO_OF_FOLDS):
                    W = initialize_weights()
                    #for j in range(NO_OF_FOLDS):
                    #    if(j != i):
                    W = neural_network(X_CV[i],Y_CV[i],iterations,learning_rate,reg_para,W)
                    temp_acc = temp_acc + calc_accuracy(X_CV[i],Y_CV[i],W,True)
                temp_acc = temp_acc/(NO_OF_FOLDS)
                CV_accuracy.append(temp_acc)
                test_acc = calc_accuracy(X_test,Y_test,W,True)
                Test_accuracy.append(test_acc)
                print("CV Accuracy = " + str(temp_acc) + " | Test Accuracy = " + str(test_acc))
                create_outputfile(iterations,learning_rate,reg_para,temp_acc,test_acc)
                Y_pred = make_predictions(X_test,W,Y_test)
                visualise(X_test,Y_test,Y_pred)
                
