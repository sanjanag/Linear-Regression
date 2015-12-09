import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import math

class LinearRegression(object):
    
    def __init__(self, fileName, ratio, reg, alpha, reg_type):

        self.ratio = ratio
        self.reg = reg
        self.alpha = alpha
        self.reg_type = reg_type
        
        self.m = 0
        self.features = 0
        
        self.read_data(fileName)

        self.testsize = int(self.ratio * self.m)
        self.trainsize = self.m - self.testsize

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split(self.X, self.Y)
        
        self.Beta = np.matrix( np.zeros((self.features+1))).getT()
        self.lr = np.identity(self.features+1)
        
        for i in range(self.features+1):
            self.lr[i,i] = self.alpha/(self.trainsize)
        
    def split(self, X, Y):

        self.testindex = self.rand_index(self.m, self.testsize)
        
        X_train = np.matrix(np.ones((self.trainsize, self.features+1)))
        X_test = np.matrix(np.ones((self.testsize, self.features+1)))
        Y_train = np.matrix(np.ones((self.trainsize))).getT()
        Y_test = np.matrix(np.ones((self.testsize))).getT()

        i_testsize = 0
        i_test = 0
        i_train = 0
        
        for i in range(self.m):
            if(i_testsize < self.testsize and i == self.testindex[i_testsize]):
                for j in range(self.features+1):
                    X_test[i_test, j] = self.X[i,j]
                Y_test[i_test, 0] = self.Y[i,0]
                i_test = i_test + 1
                i_testsize = i_testsize+1
            else:
                for j in range(self.features+1):
                    X_train[i_train, j] = self.X[i,j]
                Y_train[i_train, 0] = self.Y[i,0]
                i_train = i_train + 1
        
        return X_train, X_test, Y_train, Y_test

    def rand_index(self, m, testsize):
        l = []
        while len(l) < self.testsize:
            temp = random.randrange(0,m)
            if temp in l:
                pass
            else:
                l.append(temp)
        l.sort()
        return l
        
    def read_data(self,f):
        f = open(f,'r')
        lines = f.readlines()
        f.close()

        self.m = len(lines)
        i = 0
        len_s = 0
        for line in lines:
            s = " ".join(line.split())
            s = s.split()
            s = [ float(x) for x in s ]
            if i == 0:
                self.features = len(s)-1
                self.X = np.ones((self.m,self.features+1))
                self.Y = np.ones((self.m))
                len_s = len(s)
            for j in range(self.features):
                self.X[i,j+1] = s[j]
            self.Y[i] = s[len_s-1]
            i = i+1
        self.X = np.matrix(self.X)
        self.Y = np.matrix(self.Y).getT()
        
    def get_Grad(self):
        a = ((self.X_train*self.Beta - self.Y_train).getT())*self.X_train
        return (a*self.lr).getT()
    
    def get_rmserr(self,X,Y, size):
        return math.sqrt(((np.sum( np.square(X*self.Beta-Y), axis = 0))[0,0])/size)

    def get_cost(self, a, b , size):
        orig_term = (np.sum(np.square(a*self.Beta-b),axis = 0))[0,0]
        if self.reg_type==2:
            reg_term = (np.sum(np.square(self.Beta), axis=0))[0,0]*self.reg
        if self.reg_type ==1:
            reg_type = (np.sum(np.absolute(self.Beta), axis = 0))[0,0]*self.reg
        return (orig_term + reg_term)/size
        
    def test(self, X, Y, size):
        return self.get_rmserr(X, Y, size)

    def train(self, X, Y, size):
        
        trainErr = []
        validErr = []
        costList = []
        
        cost = self.get_cost(X, Y, size)
        prevcost = cost
        regmat = np.identity(self.features+1)
        if self.reg_type == 2:
            for i in range(1,self.features+1):
                regmat[i,i] = 1 - (self.alpha*self.reg)/self.trainsize
        if self.reg_type == 1:
            regmat = np.zeros((self.features+1))
            temp = (self.reg*self.alpha)/self.trainsize
            for i in range(1,self.features+1):
                regmat[i] = temp
            regmat = (np.matrix(regmat)).getT()
        i=0
        while (i<1000000):
            
            grad = self.get_Grad()
            if self.reg_type ==2:
                self.Beta = (regmat*self.Beta) - grad
            if self.reg_type == 1:
                self.Beta  = self.Beta - grad - regmat
            cost = self.get_cost(X, Y, size)
            costList.append(cost)
            if i%100 == 0:
                print cost

#            print len(trainErr), i
            
            trainErr.append(self.get_rmserr(X, Y, size))
            validErr.append(self.get_rmserr(self.X_test, self.Y_test, self.testsize))
            
            if(cost >= prevcost or (cost < prevcost and abs(cost - prevcost) < 0.0000001)):
                print cost , prevcost
                break
            else:
                prevcost = cost
            i= i+1
        return self.Beta, trainErr, validErr, costList, i

f = sys.argv[1]
ratio = float(sys.argv[2])
reg = float(sys.argv[3])
alpha = float(sys.argv[4])
reg_type = int(sys.argv[5])
model = LinearRegression(f,ratio,reg,alpha,reg_type)

Beta, trainErr, validErr, costList, n_epoch = model.train(model.X_train, model.Y_train ,model.trainsize)

testErr = model.test(model.X_test, model.Y_test, model.testsize)
print "Test error ", testErr
print "Train error ",  trainErr[len(trainErr)-1]

def plot_learning_curve(trainErr,validErr,costList, n_epoch):
    print "length of train_Err ", len(trainErr), "n_epoch ", n_epoch
    plt.plot(np.arange(n_epoch), np.array(validErr), np.arange(n_epoch),  np.array(trainErr), 'k--', np.arange(n_epoch),np.array(costList), 'r')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend(('Validation_Err', 'Train Error', 'Cost_function'), shadow=True, loc=(0.01, 0.55))
    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize=20, color='b')
    plt.setp(ltext[1], fontsize=20, color='g')
    plt.setp(ltext[2], fontsize = 20, color = 'r')
    plt.show()
    
plot_learning_curve(trainErr,validErr,costList, n_epoch)
