
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.kernel_ridge import KernelRidge
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


def PolyRegrPol(fulldata):

    sz = fulldata.shape[0]
   
    data = fulldata[:,4:16].astype(float)
    label = np.concatenate((fulldata[:,-6].reshape(sz,1),fulldata[:,-2].reshape(sz,1),fulldata[:,-1].reshape(sz,1)),axis=1).astype(float)
    
    x, x_test, y, y_test = train_test_split(data,label,test_size=0.11,random_state=12,shuffle=True)
    
    NO2x, NO2y, NO2x_test, NO2y_test = x[:,:9],y[:,0],x_test[:,:9],y_test[:,0]
    
    COx, COy, COx_test, COy_test = x[:,3:12],y[:,1],x_test[:,3:12],y_test[:,1]
    
    alpha_,degree = 0.001,5
    plt.plot(COy_test, color='cornflowerblue', linewidth=2,
                 label="True Val")
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_, normalize=True))

    model.fit(COx, COy)
    COy_pred = model.predict(COx_test)
    
    plt.plot(COy_pred, color='gold', linewidth=2,
             label="Est Val")
    plt.legend(loc='upper left')
    
    plt.subplots_adjust(bottom=0.3)
    
    plt.annotate('\nCO Estimation\nMSE: %.2f \nVariance Score: %.2f'  
                 % (mean_squared_error(COy_test, COy_pred),r2_score(COy_test, COy_pred)),
                xy=(0.5, 0), xytext=(20, 10),
                xycoords=('figure fraction', 'figure fraction'),
                textcoords='offset points',
                size=12, ha='center', va='bottom')
    
    plt.show()
    
    print('\n5-Order Polynomial Regression\nCO Estimation\nMSE: %.2f \nVariance Score: %.2f'  
                 % (mean_squared_error(COy_test, COy_pred),r2_score(COy_test, COy_pred)))
    
    plt.plot(NO2y_test, color='cornflowerblue', linewidth=2,
                 label="True Val")
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha_, normalize=True))
    model.fit(NO2x, NO2y)
    NO2y_pred = model.predict(NO2x_test)

    plt.plot(NO2y_pred, color='red', linewidth=2,
             label="Est Val")
    plt.legend(loc='upper left')
    
    plt.subplots_adjust(bottom=0.3)
    
    plt.annotate('\nNO2 Estimation\nMSE: %.2f \nVariance Score: %.2f'  
                 % (mean_squared_error(NO2y_test, NO2y_pred),r2_score(NO2y_test, NO2y_pred)),
                xy=(0.5, 0), xytext=(20, 10),
                xycoords=('figure fraction', 'figure fraction'),
                textcoords='offset points',
                size=12, ha='center', va='bottom')
    
    plt.show()
    
    print('\n5-Order POlynomial Regression\nNO2 Estimation\nMSE: %.2f \nVariance Score: %.2f'  
                 % (mean_squared_error(NO2y_test, NO2y_pred),r2_score(NO2y_test, NO2y_pred)))



def LinearRegrPol(data,label,pol):
    
    x_train, x_test ,y_train,y_test = train_test_split(data,label,test_size=0.11, random_state=12, shuffle=True)

    #regr = KernelRidge(kernel="linear")
    regr = linear_model.LinearRegression(normalize=False)
    regr.fit(x_train, y_train)
    
    y_pred = regr.predict(x_test)
    
    print("---LINEAR MODEL %s Estimation---" %(pol))
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
        
    #Subplots
    fig = plt.figure()
    gs = gridspec.GridSpec(6, 2)
    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[0:2,1])
    ax3 = fig.add_subplot(gs[3:,:])
    
    ax1.plot(y_test, color="xkcd:azure")
    ax1.set_title("True Values")
    ax2.plot(y_pred, color="xkcd:lightgreen")
    ax2.set_title("Estimated Values")
    
    ax3.plot(y_test, color="xkcd:azure", label="True Val")
    ax3.plot(y_pred, color="xkcd:lightgreen", label="Est Val")
    ax3.set_title("Combined")
    ax3.legend(loc='upper left')
    plt.subplots_adjust(bottom=0.25)
    
    ax3.annotate('\n%s Estimation\nMSE: %.2f \nVariance Score: %.2f'  
                 % (pol,mean_squared_error(y_test, y_pred),r2_score(y_test, y_pred)),
                xy=(0.5, 0), xytext=(20, 10),
                xycoords=('figure fraction', 'figure fraction'),
                textcoords='offset points',
                size=12, ha='center', va='bottom')
    plt.show()


def LinearRegression(fulldata):
    dataC = fulldata[:,7:16].astype(float)
    labelC = fulldata[:,-2].astype(float)
    
    LinearRegrPol(dataC,labelC,"CO")
    
    dataN = fulldata[:,4:13].astype(float)
    labelN = fulldata[:,-6].astype(float)
    
    LinearRegrPol(dataN,labelN,"NO2")
    
    
    

    