import numpy as np
import sys
import matplotlib.pyplot as plt
import math
class LogisticRegression(object):

    def __init__(self, trainfileName, testfileName, alpha):
        self.alpha = alpha
        self.features = 0
        
        self.read_data(trainfileName, testfileName)
        self.Beta = np.matrix( np.zeros((self.features+1))).getT()
    def read_data(self, testfile, trainfile):
        testfile = open(testfile, 'r')
        trainfile = open(trainfile, 'r')
        lines_train  = trainfile.readlines()
        lines_test = testfile.readlines()
        testfile.close()
        trainfile.close()
        self.trainsize  = len(lines_train)
        self.testsize = len(lines_test)
        len_s = 0
        i = 0
        for line in lines_train:
            s = " ".join(line.split(","))
            s = s.split()
            s = [int(x) for x in s]
            if i == 0:
                self.features = len(s)-1
                self.X_train = np.ones((self.trainsize, self.features+1))
                self.Y_train = np.ones((self.trainsize))
                len_s = len(s)
            for j in range(1,len_s):
                self.X_train[i, j] = s[j]
            self.Y_train[i] = s[0]
            i = i+1

        self.X_train = np.matrix(self.X_train)
        self.Y_train = np.matrix(self.Y_train).getT()
        i = 0
        for line in lines_test:
            s = " ".join(line.split(","))
            s = s.split()
            s = [int(x) for x in s]
            if i == 0:
                self.X_test = np.ones((self.testsize, self.features+1))
                self.Y_test = np.ones((self.testsize))
                len_s = len(s)
            for j in range(1,len_s):
                self.X_test[i,j] = s[j]
            self.Y_test[i] = s[0]
            i = i+1
        self.X_test = np.matrix(self.X_test)
        self.Y_test = np.matrix(self.Y_test).getT()

    def get_cost(self, a, b, size):
        c = a*self.Beta
        for i in range(c.size):
            c[i,0] = 1/(1+math.exp(c[i,0]))
        d = np.matrix(np.ones((size))).getT() - c
        b1 = np.matrix(np.ones((size))).getT() - b
        c = np.log(c)
        d = np.log(d)
        return (((b.getT()*c)[0,0] + (b1.getT()*d)[0,0])*(-1))/size

    def get_Grad(self, X, Y):
        a = X*self.Beta
        for i in range(a.size):
            a[i,0] = 1/(1+math.exp(a[i,0]))
        a = ((a-Y).getT())*X
        return (a*self.lr).getT()
        
    def train(self, X, Y, size):
        costList = []
        cost = self.get_cost(X,Y,size)
        prevcost = cost
        self.lr = np.identity(self.features+1)
        for i in range(self.features+1):
            self.lr[i,i] = self.alpha/(self.trainsize)
        i = 0
        while(i < 1000000):
            grad = self.get_Grad(X,Y)
            self.Beta = self.Beta - grad
            cost = self.get_cost(X,Y,size)
            costList.append(cost)
            if i%100 == 0:
                print cost

            i = i+1
            
            if(cost >= prevcost or (cost < prevcost and abs(cost-prevcost) < .0000001)):
                print cost, prevcost
                break
            else:
                prevcost = cost
        return self.Beta, costList, i

trainf = sys.argv[1]
testf = sys.argv[2]
alpha = float(sys.argv[3])

model = LogisticRegression(trainf, testf, alpha)

Beta, costList, n_epoch = model.train(model.X_train, model.Y_train, model.trainsize)

def plot_learning_curve(costList, n_epoch):
    print "length of costList ", len(costList), " n_epoch ", n_epoch
    plt.plot(np.arange(n_epoch), np.array(costList))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.show()

plot_learning_curve(costList, n_epoch)
