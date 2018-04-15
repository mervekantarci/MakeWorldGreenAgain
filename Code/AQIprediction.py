
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model


def AQIEstimation(fulldata):
    
    sz = fulldata.shape[0]

    aqilevel = fulldata[:,-1:].astype(int)//15
    
    fulldata = np.concatenate((fulldata,aqilevel),axis=1)

    data = fulldata[:,4:16].astype(float)
    label = np.concatenate((fulldata[:,-7].reshape(sz,1),fulldata[:,-3].reshape(sz,1),fulldata[:,-2].reshape(sz,1),aqilevel),axis=1).astype(float)
    
    x, x_test, y, y_test = train_test_split(data,label,test_size=0.11,random_state=12,shuffle=True)
       
    NO2x, NO2y, NO2x_test, NO2y_test = x[:,:9],y[:,0],x_test[:,:9],y_test[:,0]
    
    COx, COy, COx_test, COy_test = x[:,3:12],y[:,1],x_test[:,3:12],y_test[:,1]
    
    AQIx, AQIy, AQIyds, AQIy_test, AQIy_testds = y[:,:2],y[:,2],y[:,3],y_test[:,2],y_test[:,3]

    alpha_,degree = 0.001,5
    
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_, normalize=True))

    spec = model.fit(COx, COy)
    COy_pred = model.predict(COx_test)
    COy_train = model.predict(COx)
    
    print('\n5-Order Polynomial Regression\nCO Estimation\nMSE: %.2f \nScore: %.2f'  
                 % (mean_squared_error(COy_test, COy_pred),r2_score(COy_test, COy_pred)))

    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_, normalize=True))
    spec = model.fit(NO2x, NO2y)
    NO2y_pred = model.predict(NO2x_test)
    NO2y_train = model.predict(NO2x)
    
    print('\n5-Order Polynomial Regression\nNO2 Estimation\nMSE: %.2f \nScore: %.2f'  
                 % (mean_squared_error(NO2y_test, NO2y_pred),r2_score(NO2y_test, NO2y_pred)))
    
    testsz = NO2y_pred.shape[0]
    
    AQIx_test = np.concatenate((NO2y_pred.reshape(testsz,1),COy_pred.reshape(testsz,1)),axis=1)
    
    s = NO2y_train.shape[0]
    AQIx = np.concatenate((NO2y_train.reshape(s,1),COy_train.reshape(s,1)),axis=1)

    plt.plot(AQIy_test, color='cornflowerblue', linewidth=2,
                 label="True Val")
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_, normalize=True))
    spec = model.fit(AQIx, AQIy)
    AQIy_pred = model.predict(AQIx_test)

    plt.plot(AQIy_pred, color='lightgreen', linewidth=2,
             label="Est Val")
    plt.legend(loc='upper left')
    
    plt.subplots_adjust(bottom=0.3)
    
    plt.annotate('\nAQI Estimation\nMSE: %.2f \nScore: %.2f'  
                 % (mean_squared_error(AQIy_test, AQIy_pred),r2_score(AQIy_test,AQIy_pred)),
                xy=(0.5, 0), xytext=(20, 10),
                xycoords=('figure fraction', 'figure fraction'),
                textcoords='offset points',
                size=12, ha='center', va='bottom')
    
    plt.show()
    
    print(('\nAQI Estimation\nMSE: %.2f \nScore: %.2f'  
                 % (mean_squared_error(AQIy_test, AQIy_pred),r2_score(AQIy_test,AQIy_pred))))
    
    clf = svm.SVC(kernel="linear")

    clf.fit(AQIx, AQIyds)
    
    y_pred = clf.predict(AQIx_test)

    # score: 1 is perfect prediction
    print('\nSVM MODEL AQI-LEVEL PREDICTION\nAccuracy score: %.2f' % np.mean(y_pred==AQIy_testds))
