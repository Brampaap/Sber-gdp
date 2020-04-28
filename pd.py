import math
import xlrd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_white
import statsmodels.formula.api as sm
import statsmodels.api as sm1
import scipy.stats as stats
from scipy.linalg import toeplitz


def Get_Data():
    '''Taking data from excel file and returning array'''

    #Opening data file
    workbook = xlrd.open_workbook('Dataset.xlsx', on_demand = True)

    #Choosing active sheet
    sheet = workbook.sheet_by_name('Исх.Данные')

    #Creating two-dimensional empty array
    data = []
    for j in range(sheet.ncols - 1):
        d2 = []
        for i in range(sheet.nrows - 1):
            d2.append(0)
        data.append(d2)

    #Inserting data from excel to data array
    for j in range(1, sheet.ncols):
        for i in range(1, sheet.nrows):
            data[j - 1][i - 1] = sheet.cell(i, j).value

    #Return clear to use array from excel
    return data


def Regression(data):
    '''Making regressional models and returning
    parameters(result of ols, Y array, X array, resids array)
    of the better one.

    Keyword argument:
    data -- the array part
    '''
    def Graphs():
        '''Graphs:
        - Linear model
        - Exponential model
        - Test for heteroskedacity
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.suptitle('Regressional models')

        ax1.grid()
        ax1.scatter(X, Y, c = '#02194f')
        ax1.plot(X, Ythlin, c = '#039cfc', label ='Forecasting values of Y')
        ax1.set_title('Linear model')
        ax1.set_facecolor('white')
        ax1.set(xlabel = 'Х2 - Company assets', ylabel = 'Y - Gross national product')
        ax1.legend()

        ax2.grid()
        ax2.scatter(X, Y, c = '#02194f')
        ax2.plot(X, Ythexp, c = '#039cfc', label = 'Forecasting values of Y')
        ax2.set_title('Exponential model')
        ax2.set_facecolor('white')
        ax2.set(xlabel = 'Х2 - Company assets', ylabel = 'Y - Gross national product')
        ax2.legend()

        fig.set_figwidth(16)     #  width and
        fig.set_figheight(6)     #  height of "Figure"

        plt.show()

        plt.figure(2)
        plt.title('Visual test for heteroskedasticity')
        plt.grid()
        plt.xlabel('Yth - Gross national product')
        plt.ylabel('ε - Resids of exp model')
        plt.scatter(Ythexp, E, c='#02194f')

        plt.show()
        '''END OF GRAPHS'''

    #Creating new dataframe
    df = pd.DataFrame({
        'Y': data[0],
        'X1': data[1],
        'X2': data[2],
        'X3': data[3],
        'X4': data[4],
        'X5': data[5]
    })

    print('\nData table')
    print(df)
    print('\n')

    #Visualizing correlation table to exclude interrelated exog factors
    print('Correlation table')
    print(df.corr())
    print('\n')

    #We need Y and X2
    Y = df[['Y']].values
    X = df.drop(['Y','X1', 'X3', 'X4', 'X5'], axis=1).values

    Lresult = sm.ols(formula="Y ~ X2", data=df).fit()#Fit linear model
    R2linear = Lresult.rsquared                      #R squared of lin model
    Elin = Lresult.resid.values                      #Resids of lin model

    #1. Ln(Y)
    #2. Y theoretic linear
    #3. Y theoretic exponentional
    #4. exp(Eexp)
    Ylog, Ythlin, Ythexp, E = [], [], [], []

    for i in range(len(X)):
        Ylog.append(math.log(data[0][i]))

    #Dataframe for exp model
    df = pd.DataFrame({
        'Y': Ylog,      #Ln(Y)
        'X': data[2]    #X2
    })

    EXPresult = sm.ols(formula="Y ~ X", data=df).fit()#Fit exp model
    R2exp = EXPresult.rsquared                        #R squared of exp model
    Eexp = EXPresult.resid.values                     #Resids of exp model

    for i in range(len(X)):
        Ythexp.append(math.exp(Ylog[i] - Eexp[i]))    #Yth = Y - E (For i obs)
        Ythlin.append(Y[i][0] - Elin[i])              #Similarly
        E.append(math.exp(Eexp[i]))                   #Normalization resids to billions

    #Choosing better model
    print('R squared for exp = ' + str(R2exp))
    print('R squared for linear = ' + str(R2linear))

    print('[-!!!-] Linear model better than exp [-!!!-]\n')
    print(Lresult.summary())
    print('\n')
    return Lresult, Y, X, Elin


def Glayser_het(Y, X, e):
    '''Glayser test for heteroskedasticity.

    Keyword arguments:
    Y -- the array part
    X -- the array part
    e -- the array part (resirds)
    '''

    def T_stat(Y, X):
        '''Student t statistic. Returning
        test value and critical value of T-statistic.

        Keyword arguments:
        Y -- the array part
        X -- the array part
        '''

        df = pd.DataFrame({
            'Y': Y,
            'X': X,
        })

        result = sm.ols(formula="Y ~ X", data=df).fit()

        T = result.tvalues.values[1]                 #Test value
        Ttab = stats.t.ppf(1-0.025, len(newX[0]) - 2)#Critical value

        return T, Ttab

    #1. One-dementional array for X
    #2. Square root of X
    #3. 1 / X
    #4. Two-dementional array for T-test
    #5. Abs(E)
    newX, sqrtX, backX, Queue, E = [], [], [], [], []

    for i in range(len(e)):
        E.append(abs(e[i]))

    for i in range(len(X[0])):
        temp1, temp2, temp3 = [], [], []
        for j in range(len(X)):
            temp1.append(0)
            temp2.append(0)
            temp3.append(0)
        sqrtX.append(temp1)
        backX.append(temp2)
        newX.append(temp3)

    for j in range(len(X[0])):
        for i in range(len(X)):
            sqrtX[j][i] = math.sqrt(X[i][j])
            backX[j][i] = 1 / (X[i][j])
            newX[j][i] = X[i][j]

    for k in range(len(newX)):
        Queue.append(newX[k])
        Queue.append(sqrtX[k])
        Queue.append(backX[k])

    Tstat = []
    answer = 'Glayser Test[-]: Model has no heteroskedacity'
    res = '-'
    for k in range(len(Queue)):
        T, Ttab = T_stat(E, Queue[k])
        if(T > Ttab):
            answer = 'Glayser Test[+]: Model has heteroskedacity'
            Tstat.append(T)
            res = '+'
    print(answer)

    for i in range(len(Tstat)):
        print(str(i + 1) + ': Tstat > Ttab ===> ' + str(Tstat[i]) + ' > ' + str(Ttab))
    print('\n')
    return res


def White_het(r, e):
    '''White test for heteroskedasticity

    Keyword arguments:
    r -- the array part (Result model of ols)
    e -- the array part (Resids of this model)
    '''

    hw = het_white(e, r.model.exog)          #White test
    tblv = stats.chi2.ppf(0.95, (len(e) - 2))#Critical value of chi squared

    if(hw[1] > 0.05):
        print('White Test[+]: Model has heteroskedacity\nPvalue Wtest = ' + str(hw[1]) + ' > 0.05\n')
        return '+'
    else:
        print('White Test[-]: Model has no heteroskedacity\nWtest = ' + str(hw[1]) + ' < 0.05\n')
        return '-'


def AddictionalTS(Yth):
    '''Additional time series model

    Keyword params:
    Yth -- the array part
    '''
    def Graphs(t, Yth, TplusS):
        '''Graphs:
        - Additional model
        '''

        plt.figure()
        plt.title('Additional model')
        plt.grid()
        plt.xlabel('t - number of quarter')
        plt.ylabel('Y - Gross national product')
        plt.scatter(t, Yth, c='#030200')
        plt.plot(t, Yth, ":", c='#030200', label = 'Real values of Y')
        plt.scatter(t, TplusS, c = "#00efac")
        plt.plot(t, TplusS, c='#039cfc', label = 'Forecasting values of Y')
        plt.legend()

        plt.show()

        '''END GRAPHS'''

    #1. Total for four quarters
    #2. Four-Quarter Moving Averages
    #3. Centered Moving Average
    #4. Seasonal component rating
    #5. Average estimate of the seasonal component for the i-th quarter
    #6. Adjusted Seasonal Component
    Sum4quart, Mean4quart, CenterM4Q, EstimSeasComp, MeanESC, CorrectSC = [], [], [], [], [], []
    AmountQuart = [0, 0, 0, 0] #Total for the i-th quarter

    for i in range(len(Yth) - 3):
        Sum4quart.append(Yth[i] + Yth[i + 1] + Yth[i + 2] + Yth[i + 3])
        Mean4quart.append((Yth[i] + Yth[i + 1] + Yth[i + 2] + Yth[i + 3]) / 4)

    for i in range(len(Mean4quart) - 1):
        CenterM4Q.append((Mean4quart[i] + Mean4quart[i + 1]) / 2)
        EstimSeasComp.append(Yth[i + 2] - CenterM4Q[i])

    for i in range(2):
        EstimSeasComp.insert(0, 0)
        EstimSeasComp.append(0)

    count = 1
    year = 1
    for i in range(len(Yth)):
        AmountQuart[count - 1] += EstimSeasComp[i]
        if (count / 4 == 1):
            year += 1
            count = 1
            continue
        count += 1

    for i in range(len(AmountQuart)):
        MeanESC.append(AmountQuart[i] / year)


    correct = 0
    for i in range(len(MeanESC)):
        correct += MeanESC[i]
    correct /= 4

    for i in range(len(MeanESC)):
        CorrectSC.append(MeanESC[i] - correct)

    sum = 0
    for i in range(len(CorrectSC)):
        sum += CorrectSC[i]
    # print(sum)                    Must be 0 for additive model

    S = []
    count = 0
    for i in range(len(Yth)):
        S.append(CorrectSC[count])
        count += 1
        if(count == 4):
            count = 0

    YminusS, T, t, TplusS, sumE2, YthMinusMeanY = [], [], [], [], 0, 0
    for i in range(len(Yth)):
        t.append(i + 1)              #Number of quarter
        YminusS.append(Yth[i] - S[i])#Theoretical Y minus season component for the quarter

    df = pd.DataFrame({
        't': YminusS,
        'YthS': t,
    })
    a = b = R2 = 0
    result = sm.ols(formula="t ~ YthS", data=df).fit()
    a = result.params.values[0]
    b = result.params.values[1]
    for i in range(len(Yth)):
        T.append(a + b * t[i])
        TplusS.append(T[i] + S[i])
        sumE2 += (Yth[i] - TplusS[i]) ** 2
        YthMinusMeanY += (Yth[i] - np.mean(Yth)) ** 2
    R2 = round(1 - sumE2 / YthMinusMeanY, 2)
    # Graphs(t, Yth, TplusS)
    print('R-squared for time series = ' + str(R2))
    for i in range(1, 6, 1):
        print('Forecast for the quarter ' + str(len(Yth) + i) + ' = ' + str(a + b * (len(Yth) + i) + S[i - 1]))


def MultiplyTS(Yth):
    '''Additional time series model

    Keyword params:
    Yth -- the array part
    '''
    def Graphs(t, Yth, TmS):
        '''Graphs:
        - Additional model
        '''

        plt.figure()
        plt.title('Multiply model')
        plt.grid()
        plt.xlabel('t - number of quarter')
        plt.ylabel('Y - Gross national product')
        plt.scatter(t, Yth, c='#030200')
        plt.plot(t, Yth, ":", c='#030200', label = 'Real values of Y')
        plt.scatter(t, TmS, c = "#00efac")
        plt.plot(t, TmS, c='#039cfc', label = 'Forecasting values of Y')
        plt.legend()

        plt.show()

        '''END GRAPHS'''
    #1. Total for four quarters
    #2. Four-Quarter Moving Averages
    #3. Centered Moving Average
    #4. Seasonal component rating
    #5. Average estimate of the seasonal component for the i-th quarter
    #6. Adjusted Seasonal Component
    Sum4quart, Mean4quart, CenterM4Q, EstimSeasComp, MeanESC, CorrectSC = [], [], [], [], [], []
    AmountQuart = [0, 0, 0, 0] #Total for the i-th quarter

    for i in range(len(Yth) - 3):
        Sum4quart.append(Yth[i] + Yth[i + 1] + Yth[i + 2] + Yth[i + 3])
        Mean4quart.append((Yth[i] + Yth[i + 1] + Yth[i + 2] + Yth[i + 3]) / 4)

    for i in range(len(Mean4quart) - 1):
        CenterM4Q.append((Mean4quart[i] + Mean4quart[i + 1]) / 2)
        EstimSeasComp.append(Yth[i + 2] / CenterM4Q[i])

    for i in range(2):
        EstimSeasComp.insert(0, 0)
        EstimSeasComp.append(0)

    count = 1
    year = 1
    for i in range(len(Yth)):
        AmountQuart[count - 1] += EstimSeasComp[i]
        if (count / 4 == 1):
            year += 1
            count = 1
            continue

        count += 1

    for i in range(len(AmountQuart)):
        MeanESC.append(AmountQuart[i] / year)

    correct = 0
    for i in range(len(MeanESC)):
        correct += MeanESC[i]
    correct = 4 / correct

    for i in range(len(MeanESC)):
        CorrectSC.append(MeanESC[i] * correct)

    S = []
    count = 0
    for i in range(len(Yth)):
        S.append(CorrectSC[count])
        count += 1
        if(count == 4):
            count = 0

    YdivS, T, t, TmS, sumE2, YthMinusMeanY = [], [], [], [], 0, 0
    for i in range(len(Yth)):
        t.append(i + 1)              #Number of quarter
        YdivS.append(Yth[i] / S[i])  #Theoretical Y minus season component for the quarter

    df = pd.DataFrame({
        't': t,
        'YthS': YdivS,
    })
    a = b = R2 = 0
    result = sm.ols(formula="YthS ~ t", data=df).fit()
    a = result.params.values[0]
    b = result.params.values[1]
    for i in range(len(Yth)):
        T.append(a + b * t[i])
        TmS.append(T[i] * S[i])
        sumE2 += (Yth[i] - TmS[i]) ** 2
        YthMinusMeanY += (Yth[i] - np.mean(Yth)) ** 2
    R2 = round(1 - sumE2 / YthMinusMeanY, 2)
    # Graphs(t, Yth, TmS)
    print('R-squared for time series = ' + str(R2))
    for i in range(1, 6, 1):
        print('Forecast for the quarter ' + str(len(Yth) + i -1) + ' = ' + str((a + b * (len(Yth) + i-1)) * S[i - 1]))


def GLS(Y, X, e, r):

    res_fit = sm1.OLS(e[1:], e[:-1]).fit()
    rho = res_fit.params
    order = toeplitz(np.arange(len(X)))
    sigma = rho**order
    gls_model = sm1.GLS(Y, X, sigma=sigma)
    gls_results = gls_model.fit()
    print(gls_results.summary())
    E = gls_results.resid

    return E


data = Get_Data()

print('[-!!!-] Time series model [-!!!-]\n')

print('Multiply Time Series')
forecast = MultiplyTS(data[0])
print('\n')
print('Addictional Time Series')
forecast = AddictionalTS(data[0])

r, Y, X, e = Regression(data)

resW = White_het(r, e)
resG = Glayser_het(Y, X, e)

if (resW == '+' or resG == '+'):
    gls_e = GLS(Y, X, e, r)
    print('\n')
