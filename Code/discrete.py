from os import system
from sklearn import tree
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

def NeuralNetwork(fulldata):
    data = fulldata.T[4:13]
    data = data.astype(float)
    label = fulldata.T[16:]
    label = label.astype(float)


    X_train, X_test, y_train, y_test = train_test_split(data.T, label.T, test_size=0.12, random_state=35)

    ytrainLabel = y_train.T[3]
    ytestLabel = y_test.T[3].reshape(-1, 1)
    clf = MLPClassifier(activation=("logistic"), alpha=1e-5, hidden_layer_sizes=(22, 12, 6), random_state=1)
    clf.fit(X_train, ytrainLabel)

    prediction = clf.predict(X_test).reshape(-1, 1)
    print("4 Labels Neural Network")
    print(classification_report(prediction, ytestLabel))

    ytrainLabel = y_train.T[2]
    ytestLabel = y_test.T[2].reshape(-1, 1)
    clf = MLPClassifier(activation=("logistic"), alpha=1e-5, hidden_layer_sizes=(22, 12, 6), random_state=1)
    clf.fit(X_train, ytrainLabel)

    prediction = clf.predict(X_test).reshape(-1, 1)
    print("6 Labels Neural Network")
    print(classification_report(prediction, ytestLabel))
    ytrainLabel = y_train.T[1]
    ytestLabel = y_test.T[1].reshape(-1, 1)
    clf = MLPClassifier(activation=("logistic"), alpha=1e-5, hidden_layer_sizes=(22, 12, 6), random_state=1)
    clf.fit(X_train, ytrainLabel)

    prediction = clf.predict(X_test).reshape(-1, 1)
    print("12 Labels Neural Network")
    print(classification_report(prediction, ytestLabel))

def DecisionTree(fulldata):
    data = fulldata.T[4:13]
    data = data.astype(float)
    label = fulldata.T[16:]
    label = label.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(data.T, label.T, test_size=0.12, random_state=35)
    ytrainLabel = y_train.T[3]
    ytestLabel = y_test.T[3].reshape(-1, 1)
    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(X_train, ytrainLabel)
    prediction = clf.predict(X_test).reshape(-1, 1)
    print("4 Labels Decision Tree")
    print(classification_report(prediction, ytestLabel))
    ytrainLabel = y_train.T[2]
    ytestLabel = y_test.T[2].reshape(-1, 1)
    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(X_train, ytrainLabel)
    prediction = clf.predict(X_test).reshape(-1, 1)
    print("6 Labels Decision Tree")
    print(classification_report(prediction, ytestLabel))
    ytrainLabel = y_train.T[1]
    ytestLabel = y_test.T[1].reshape(-1, 1)
    clf = tree.DecisionTreeClassifier(max_depth=6)
    clf = clf.fit(X_train, ytrainLabel)
    prediction = clf.predict(X_test).reshape(-1, 1)
    print("12 Labels Decision Tree")
    print(classification_report(prediction, ytestLabel))