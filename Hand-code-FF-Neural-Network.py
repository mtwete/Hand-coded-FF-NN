"""
Matthew Twete

A hand-coded feedforward neural network trained on MNIST classification
using momentum in backpropogation. Experiments included using various numbers of
hidden neurons, momentum values, and different amounts of training data.

"""



#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time



#Total number of training data
Ndata = 60000


#Read in the data

#Number of classes and the shape of the input data
num_classes = 10
input_shape = (28, 28, 1)

#Get the training data (x_train), training labels (y_train), test data (x_test)
#and test labels (y_test) from the keras library
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Normalize the data so it is between 0 and 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
#Shape the training and test data so they are the same as input_shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#Reshape the training and test data so they are a (1,784)
x_test = x_test.reshape(10000,784)
x_train = x_train.reshape(60000,784)

#Add the bias to the training and test images
x_test = np.concatenate((np.ones((10000,1)), x_test),axis=1)
x_train = np.concatenate((np.ones((60000,1)),x_train),axis=1)

#Get the training and test labels as one hot encoded vectors 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Change labels from one hot encoded to sigmoid activation values
y_train = np.where(y_train == 1, 0.9, 0.1)
y_test = np.where(y_test == 1, 0.9, 0.1)


#Class for creating a single hidden layer multilayer perceptron For Mnist Data
#with variable number of hidden neurons and momentum values
class mnistmlp:
    #Single Hidden Layer Multilayer Perceptron For Mnist Data constructor
    #numHidden is the number of hidden neurons
    #numdata is the number of input data
    #momen is the momentum to be used
    def __init__(self, numHidden, numdata, momen):
        #Number of input nodes including bias
        self.inNodes = 785
        #Number of output neurons
        self.outNodes = 10
        #Number of input data
        self.numData = numdata
        #Momentum
        self.momentum = momen
        #Eta, learning rate
        self.eta = 0.1
        #Number of hidden neurons
        self.numHidden = numHidden
        #Number of epochs to train
        self.nEpoch = 50
        #Set up hidden weights array
        self.hWeights = np.random.uniform(-0.05,0.05,(self.inNodes,self.numHidden))
        #Set up output weights array
        self.oWeights = np.random.uniform(-0.05,0.05,(self.numHidden+1,self.outNodes))
        #Array to hold training data accuracy for each epoch
        self.trainAcc = np.zeros(self.nEpoch)
        #Array to hold test data accuracy for each epoch
        self.testAcc = np.zeros(self.nEpoch)
        #Array for the confusion matrix
        self.conmat = np.zeros((self.outNodes+1,self.outNodes+1))
    
    #Function to fill in and display the confusion matrix
    #output is the activations for all of the test data
    #label is the test data labels
    def conf_matrix(self, output, label):
        #Fill in the class labels in the confusion matrix
        for i in range(10):
            self.conmat[i+1][0] = i
            self.conmat[0][i+1] = i
        #Fill in the confusion matrix and print with scientific notation off
        for i in range(np.shape(output)[0]):
            self.conmat[np.argmax(label[i])+1][np.argmax(output[i])+1] += 1
        np.set_printoptions(suppress=True)
        print(self.conmat)    
        
    #Function to randomize the order of a data and label array
    #x and y are the arrays to be randomized
    def randomize_inputs(self,x,y):
        #Make sure they are the same length
        if (len(x) != len(y)):
            return
        #Get a random order and return the shuffled data and label arrays
        order = np.random.permutation(len(x))
        return x[order],y[order]
    
    #Sigmoid function
    #x is the input to the sigmoid function
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    #Calculate the test and training data accuracy for a given epoch
    #trainOut is the training data output values
    #trainLabel is the training data labels
    #testOut is the test data output values
    #testLabel is the test data labels
    #epoch is the epoch which the accuracy is being calculated for
    def calc_acc(self, trainOut, trainLabel, testOut, testLabel, epoch):
        #Set up counters and constants for calculations
        trainCorrect = 0
        trainSize = self.numData
        testCorrect = 0
        testSize = np.shape(testLabel)[0]
        #Calculate training accuracy and store the value
        for i in range(trainSize):
            index = np.argmax(trainOut[i])
            if (trainLabel[i][index] == 0.9):
                trainCorrect += 1
        self.trainAcc[epoch] = trainCorrect/trainSize
        #Calculate training accuracy and store the value
        for i in range(testSize):
            index = np.argmax(testOut[i])
            if (testLabel[i][index] == 0.9):
                testCorrect += 1
        self.testAcc[epoch] = testCorrect/testSize

    #Forward propogation, for use when propogating a single data point
    #inputs is the single data point to forward propogate
    def single_forward(self, inputs):
        #Calculate hidden activation, need to use self because it 
        #will need these values for weight updates
        self.hidAct = np.dot(inputs,self.hWeights)
        #Run through sigmoid
        self.hidAct = self.sigmoid(self.hidAct)
        #Add bias
        self.hidAct = np.concatenate((np.ones(1),self.hidAct))
        #Calculate output activation
        outAct = np.dot(self.hidAct, self.oWeights)
        #Return after running through sigmoid
        return self.sigmoid(outAct)
    
    #Forward propogation, for use when propogating all data points
    #inputs is all of the data to forward propogate
    def forward(self, inputs):
        #Calculate hidden activation
        self.hidAct = np.dot(inputs,self.hWeights)
        #Run through sigmoid
        self.hidAct = self.sigmoid(self.hidAct)
        #Add bias
        self.hidAct = np.concatenate((np.ones((np.shape(inputs)[0],1)),self.hidAct),axis=1)
        #Calculate output activation
        outAct = np.dot(self.hidAct, self.oWeights)
        #Return after running through sigmoid
        return self.sigmoid(outAct)
    
    #Train the MLP on the training data, calculate accuracy after each epoch
    #and fill in and display the confusion matrix on the test data after training
    #train is the training data
    #trainLabel is the training data labels
    #test is the test data
    #testLabel is the test data labels
    def train(self, train, trainLabel, test, testLabel):
        #Array to hold the change (capital delta) in hidden weights
        changeHidWeight = np.zeros((np.shape(self.hWeights)))
        #Array to hold the change (capital delta) in output weights
        changeOutWeight = np.zeros((np.shape(self.oWeights)))
        #Get the start time of training, to be used for timing
        t0 = time.time()
        for i in range(self.nEpoch):
            #Compute and print the estimated time it will take to train (just for my knowledge)
            if(i == 1):
                t1 = time.time()
                print("estimated time to finish: ",(t1-t0)*50/60," minutes.")
            #Randomize the order of the training data so the MLP doesn't learn the order
            train, trainLabel = self.randomize_inputs(train,trainLabel)
            #Loop over the data
            for j in range(self.numData):
                #Get the output (and hidden) activation for training datum j
                outAct = self.single_forward(train[j])
                #Calculate delta (lower case) value for the output neurons
                delOut = outAct*(1-outAct)*(trainLabel[j]-outAct)
                #Calculate the weight updates for the output layer
                changeOutWeight = self.eta*np.dot(self.hidAct.reshape(-1,1),np.transpose(delOut.reshape(-1,1))) + self.momentum*changeOutWeight
                #Calculate delta (lower case) value for the hidden neurons
                delHid = self.hidAct*(1-self.hidAct)*np.dot(self.oWeights,np.transpose(delOut))
                #Get rid of delHid value for the bias and reshape
                delHid = delHid[1:self.numHidden+1]
                delHid = np.transpose(delHid.reshape(-1,1))
                #Calculate the weight updates for the hidden layer
                changeHidWeight = self.eta*(np.dot(train[j].reshape(-1,1),delHid)) + self.momentum*changeHidWeight
                #Update weights
                self.oWeights += changeOutWeight
                self.hWeights += changeHidWeight
            #Get test data output activation
            testOut = self.forward(test)
            #Get training data output activation
            trainOut = self.forward(train)
            #Calculate accuracies
            self.calc_acc(trainOut,trainLabel,testOut,testLabel,i)
        #Get test data output activation for the confusion matrix now training is done
        testOut = self.forward(test)
        #Compute and print the time it took to train (just for my knowledge)
        t2 = time.time()
        print("Time to train network: ",(t2-t0)/60," minutes.")
        #Fill in and print the confusion matrix
        self.conf_matrix(testOut, testLabel)
    
    #Get the minimum and maximum accuracy values (with some padding) between the test 
    #and training data to be used when plotting to set the y-axis range
    def min_max(self):
        #Get the minimum accurarcy, subtract 0.02 to make the plot look 
        #better and make sure it is not less than 0
        amin = np.amin(self.testAcc)
        if (np.amin(self.trainAcc) < amin):
            amin = np.amin(self.trainAcc)
        if (amin - 0.02 < 0):
            amin = 0
        else:
            amin -= 0.02
        #Get the maximum accurarcy, add 0.02 to make the plot look 
        #better and make sure it is not more than 1
        amax = np.amax(self.testAcc)
        if (np.amax(self.trainAcc) > amax):
            amax = np.amax(self.trainAcc)
        if (amax + 0.02 > 1):
            amax = 1
        else:
            amax += 0.02
        return amin,amax
        
    #Plot accuracy of test and training data for differing number of hidden nodes and momenta 
    #after training on all of the data
    def plot_acc1(self):
        #Set up epoch values for x-axis
        epochs = np.arange(1,self.nEpoch + 1)
        #Plot the test and training accuracies
        plt.plot(epochs, self.testAcc, 'b-', label='Test data accuracy')
        plt.plot(epochs, self.trainAcc, 'g-', label='Training data accuracy')
        #Get the min and max values for the y-axis
        amin, amax = self.min_max()
        #Set y-axis range
        plt.ylim(amin,amax)
        #Add text to the plot and display
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy plot for training on all data with " + str(self.numHidden) + " hidden nodes and momentum = " + str(self.momentum) + ".")
        plt.show()
    
    #Plot accuracy of test and training data after training on one quarter of the training examples
    def plot_acc2(self):
        #Set up epoch values for x-axis
        epochs = np.arange(1,self.nEpoch + 1)
        #Plot the test and training accuracies
        plt.plot(epochs, self.testAcc, 'b-', label='Test data accuracy')
        plt.plot(epochs, self.trainAcc, 'g-', label='Training data accuracy')
        #Get the min and max values for the y-axis
        amin, amax = self.min_max()
        #Set y-axis range
        plt.ylim(amin,amax)
        #Add text to the plot and display
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy plot for training on one quarter of the data with 100 hidden nodes and momentum = 0.9.")
        plt.show()
    
    #Plot accuracy of test and training data after training on one half of the training examples
    def plot_acc3(self):
        #Set up epoch values for x-axis
        epochs = np.arange(1,self.nEpoch + 1)
        #Plot the test and training accuracies
        plt.plot(epochs, self.testAcc, 'b-', label='Test data accuracy')
        plt.plot(epochs, self.trainAcc, 'g-', label='Training data accuracy')
        #Get the min and max values for the y-axis
        amin, amax = self.min_max()
        #Set y-axis range
        plt.ylim(amin,amax)
        #Add text to the plot and display
        plt.legend(loc='upper left')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy plot for training on one half of the data with 100 hidden nodes and momentum = 0.9.")
        plt.show()




        

#Experiment 1, train MLPs on all of the data with 
#20, 50 and 100 hidden units and momentum set to 0.9
MLP1 = mnistmlp(20,Ndata,0.9)
MLP1.train(x_train,y_train,x_test,y_test)
MLP1.plot_acc1()
MLP2 = mnistmlp(50,Ndata,0.9)
MLP2.train(x_train,y_train,x_test,y_test)
MLP2.plot_acc1()
MLP3 = mnistmlp(100,Ndata,0.9)
MLP3.train(x_train,y_train,x_test,y_test)
MLP3.plot_acc1()

#Experiment 2, train MLPs on all of the data with 
#momentum set to 0.0, 0.25 and 0.5 with 100 hidden nodes
MLP4 = mnistmlp(100,Ndata,0.0)
MLP4.train(x_train,y_train,x_test,y_test)
MLP4.plot_acc1()
MLP5 = mnistmlp(100,Ndata,0.25)
MLP5.train(x_train,y_train,x_test,y_test)
MLP5.plot_acc1()
MLP6 = mnistmlp(100,Ndata,0.5)
MLP6.train(x_train,y_train,x_test,y_test)
MLP6.plot_acc1()


#Experiment 3, train MLPs on one half and one quarter of the data 
#with 100 hidden units and momentum set to 0.9

#Shuffle the training data randomly to make sure the data roughly 
#balanced amoung the 10 different classes
order = np.random.permutation(len(x_train))

#Select half of the randomized data and train an MLP on it
halfdata = x_train[order]
halflabel = y_train[order]
halfdata = halfdata[0:int(Ndata/2)]
halflabel = halflabel[0:int(Ndata/2)]
MLP7 = mnistmlp(100,int(Ndata/2),0.9)
MLP7.train(halfdata,halflabel,x_test,y_test)
MLP7.plot_acc3()

#Select a quarter of the randomized data and train an MLP on it
quardata = x_train[order]
quarlabel = y_train[order]
quardata = quardata[int(Ndata/2):int((3/4)*Ndata)]
quarlabel = quarlabel[int(Ndata/2):int((3/4)*Ndata)]
MLP8 = mnistmlp(100,int(Ndata/4),0.9)
MLP8.train(quardata,quarlabel,x_test,y_test)
MLP8.plot_acc2()




