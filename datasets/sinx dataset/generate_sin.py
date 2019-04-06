from math import sin, pi
from random import uniform

X = []

for i in range(10000):
    s = uniform(0, 2*pi)
    X.append([sin(s), s])

X = [str(i) + ',' + str(j) + '\n' for i, j in X]

fh = open('sinx.csv', 'w')
fh.writelines(X)
fh.close()

