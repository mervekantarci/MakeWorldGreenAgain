"""
BBM406 - FUNDAMENTALS OF MACHINE LEANRING


MAKE WORLD GREEN AGAIN!

Ihsan Baran Sonmez
Merve Gul Kantarcı
Derya Ersoy

"""

import numpy as np
import argparse
import continuous as regr
import discrete as dsc
import AQIprediction as aqi

#ARGS
parser = argparse.ArgumentParser(description='MAKE WORLD GREEN AGAIN PROJECT')
parser.add_argument('--classify', help="If this flag is used, classification results are displayed, default classifier is decision tree",
                    action='store_true')
parser.add_argument('--neural', action = 'store_true',
                    help="If this flag is used with --class, instead of decision tree, neural network is displayed")
parser.add_argument('--tree', action = 'store_true',
                    help="If this flag is used with --class, instead of neural network, decisin tree is displayed")
parser.add_argument('--peakvalue', help="If this flag is used, regression results are displayed, default polynomial regression)"
                   , action='store_true')
parser.add_argument('--linear', help="If this flag is used with peakvalue flag, linear regression results are displayed)"
                   , action='store_true')
parser.add_argument('--poly', help="If this flag is used with peakvalue flag, polynomial regression results are displayed)"
                   , action='store_true')
parser.add_argument('--aqi', help="If this flag is used aqi estimation is displayed"
                   , action='store_true')
args = parser.parse_args()

dataset = np.load('data-airpol.npy')

printed = False

if(args.classify):
    printed = True
    if (args.neural):
        dsc.NeuralNetwork(dataset)
    else:
        dsc.DecisionTree(dataset)

if (args.peakvalue):
    printed = True
    if (args.linear):
        regr.LinearRegression(dataset)
    else:
        regr.PolyRegrPol(dataset)

if((not printed) or args.aqi):
    aqi.AQIEstimation(dataset)









