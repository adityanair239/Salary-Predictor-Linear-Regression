from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


def coef(x,y,mx,my):
    n = np.size(x)
    SSxx = np.sum(x*x)-n*mx*mx
    SSyx = np.sum(y*x)-n*mx*my
    b1 = SSyx/SSxx
    b0 = my - b1*mx
    return b0,b1

def plot_line(x,x_test,y_test,b):
    plt.scatter(x_test, y_test, color="b",marker="o", s=30)
    y_pred = b[0] + b[1]*x
    plt.plot(x,y_pred)
    plt.savefig('graph.png')
    plt.xlabel('Years of Expirience')
    plt.ylabel('Salary')
    plt.show()
    plt.savefig('graph.png')


def main():
    data = pd.read_csv('Data/Salary_Data.csv')
    x = data['YearsExperience']
    y = data['Salary']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    mx = np.mean(x_train)
    my = np.mean(y_train)
    b = coef(x_train,y_train,mx,my)
    plot_line(x_train,x_test,y_test,b)


main()

