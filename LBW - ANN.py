#!/usr/bin/env python
# coding: utf-8

# **Machine Intelligence Assignment**

# Model specifications
# 
# Number of layers = 2
# Number of input features(neurons) = 12
# Number of Neurons in hidden layer1 = 16
# Number of Neurons in hidden layer2 = 8
# Number of Neurons in output layer = 1
# Weights Initialization : Xavier Initialization to avoid vanishing gradients from sigmoid
# Activation function in all layers = Sigmoid
# Learning rate is updated using Adam optimizer
# 
# The model with the best validation accuracy is chosen and the corresponding epochs trained and training accuracy is printed for those weights

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer


df = pd.read_csv('LBW_Dataset.csv')
imputer = KNNImputer(n_neighbors=3, weights="uniform")
df1 = pd.DataFrame(imputer.fit_transform(df))
df1.columns = list(df.columns)
df2 = pd.get_dummies(df1, columns=["Community"], prefix="Community")
scaler = preprocessing.MinMaxScaler()
df3 = pd.DataFrame(scaler.fit_transform(df2.values))
df3.columns = list(df2.columns)
df = df3

class NN:
    def __init__(self):
        self.layers = [(12, 16), (16, 8), (8, 1)]

    # Initialization of weights for each layers using Xavier intialization
    def init_layers(self, seed=66):
        param = {}
        for idx, layer in enumerate(self.layers):
            layer_no = idx + 1
            input_size = layer[0]
            output_size = layer[1]
            param['W' + str(layer_no)] = np.random.randn(
                output_size, input_size) * np.sqrt(1/(output_size+input_size))
        return param
    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))
    def sigmoid_derivative(self, dA, Z):
        s = self.sigmoid(Z)
        return dA * s * (1 - s)

    # multiplying weights by the input neurons for each layers and passing it through a sigmoid function
    def forward_prop(self, X, param):
        cache = {}
        A_curr = X
        for idx in range(len(self.layers)):
            layer_no = idx + 1
            A_prev = A_curr
            W_curr = param["W" + str(layer_no)]
            Z_curr = np.dot(W_curr, A_prev)
            A_curr = self.sigmoid(Z_curr)
            cache["A" + str(idx)] = A_prev
            cache["Z" + str(layer_no)] = Z_curr
        return A_curr, cache

    # Binary Cross Entropy loss is calculated for every prediction Y_hat
    # Adding a small value (0.001) to avoid 'RuntimeWarning: divide by zero encountered in log'
    def loss_function(self, Y_hat, Y):
        n = Y_hat.shape[1]
        cost = -1 / n * (np.dot(Y, np.log(Y_hat + 0.01).T) +
                         np.dot(1 - Y, np.log(1 - Y_hat + 0.01).T))
        return np.squeeze(cost)

    # derivative of weights is found using chain rule for every layer
    def backward_prop(self, Y_hat, Y, memory, param):
        grad_val = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
        dA_prev = - (np.divide(Y, Y_hat + 0.01) -
                     np.divide(1 - Y, 1 - Y_hat + 0.01))
        for layer_idx_prev, layer in reversed(list(enumerate(self.layers))):
            layer_idx_curr = layer_idx_prev + 1
            dA_curr = dA_prev
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = param["W" + str(layer_idx_curr)]
            dZ_curr = self.sigmoid_derivative(dA_curr, Z_curr)
            dW_curr = np.dot(dZ_curr, A_prev.T) / A_prev.shape[1]
            dA_prev = np.dot(W_curr.T, dZ_curr)
            grad_val["dW" + str(layer_idx_curr)] = dW_curr
        return grad_val

    # weights are updated for each layer using adam optimizer
    def update_weights(self, param, grads_values, learning_rate, prev_grad, alpha):
        for layer_number in range(1, len(self.layers) + 1):
            if "W" + str(layer_number) not in alpha:
                alpha["W" + str(layer_number)] = np.zeros(
                    param["W" + str(layer_number)].shape)+0.01
            alpha["W" + str(layer_number)] = np.sqrt(0.6*np.square(grads_values["dW" +
                                                                                str(layer_number)])+0.4*np.square(alpha["W" + str(layer_number)]))
            if "W" + str(layer_number) not in prev_grad:
                prev_grad["W" + str(layer_number)] = np.zeros(
                    param["W" + str(layer_number)].shape)
            param["W" + str(layer_number)] -= (learning_rate/(alpha["W" + str(layer_number)]+0.01)) * (
                0.6*grads_values["dW" + str(layer_number)]+0.4*prev_grad["W" + str(layer_number)])
            prev_grad["W" + str(layer_number)] = 0.6*grads_values["dW" +
                                                            str(layer_number)]+0.4*prev_grad["W" + str(layer_number)]
        return param
    def fit(self, X, Y, epochs):
        param = self.init_layers(self.layers)
        loss_hist = []
        learning_rate = 0.4
        prev_grad = {}
        alpha = {}
        while(epochs):
            Y_hat, cache = self.forward_prop(X, param)
            loss = self.loss_function(Y_hat, Y)
            loss_hist.append(float(loss))
            grad_val = self.backward_prop(
                Y_hat, Y, cache, param)
            param = self.update_weights(
                param, grad_val, learning_rate, prev_grad, alpha)
            epochs -= 1
        self.param = param
        return param, loss_hist
    def threshold(self, y_pred):
        for i in range(len(y_pred)):
            y_pred[i] = 1 if y_pred[i] > 0.6 else 0
    def predict(self, X):
        yhat, _ = self.forward_prop(X, self.param)
        self.threshold(yhat[0])
        return yhat[0]
    def CM(self, y_test, y_test_obs):
        for i in range(len(y_test_obs)):
            y_test_obs[i] = 1 if y_test_obs[i] > 0.6 else 0
        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp+1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn+1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp+1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn+1
        # print(tp,tn,fp,fn)
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)
        print("Confusion Matrix : ")
        for i in cm:
            print(i)
        print("\n")
        print(f"Precision : {p*100:.3f} %")
        print(f"Recall : {r*100:.3f} %")
        print(f"F1 SCORE : {f1*100:.3f} %")

model = NN() # creating model object

Y = df.Result
X = df.drop(['Result'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=75)


# The checkpoints of each model (weights, accuracy, epochs) are stored in all_models dictionary
train = []
test = []
all_models = {}


# This kindoff mimics keras-tuner, which is used to determine the optimal set of hyperparameters for the model.

for i in range(500, 5000, 100):
    param, loss_hist = model.fit(
        np.array(X_train.T), np.array(Y_train.T).reshape(1, Y_train.shape[0]), i)
    yhat = model.predict(np.array(X_train.T))
    train_accuracy = accuracy_score(Y_train.T, yhat)
    train.append(train_accuracy)
    yhat_ = model.predict(np.array(X_test.T))
    val_accuracy = accuracy_score(Y_test.T, yhat_)
    test.append(val_accuracy)
    all_models[val_accuracy] = [param, train_accuracy, yhat_, i]

best_model = sorted(all_models.items(), reverse=True)[0]

print("Optimum Epochs : ",best_model[1][3])
print(f"Training_Accuracy : {best_model[1][1]*100:.3f} %")
print(f"Validation_Accuracy : {best_model[0]*100:.3f} %\n")
model.CM(Y_test.T.ravel(), best_model[1][2])



