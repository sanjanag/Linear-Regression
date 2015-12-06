import numpy as np
import sys
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
import random

class LinearRegression(object):
    
    def __init__(self,fileName, n, ratio):
        
        self.n = n
        self.m = 0
        self.ratio = ratio
    
    
        self.X, self.Y= self.read_data(fileName)
        self.testsize = int(self.ratio * self.m)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split(self.X, self.Y)
       # print self.Y_test
        self.alpha = 0.000006999999
        self.Beta = np.matrix( np.zeros((n+1))).getT()
        self.lr = np.identity(n+1)
        for i in range(n+1):
            self.lr[i,i] = self.alpha/self.m


    def split(self, X, Y):

        self.testindex = self.rand_index(self.m, self.testsize)
        
       # print self.testsize, self.testindex
        X_train = np.ones((self.m - self.testsize, self.n+1))
        X_test = np.ones((self.testsize, self.n+1))
        Y_train = np.ones((self.m - self.testsize))
        Y_test = np.ones((self.testsize))
        
        X_train = np.matrix(X_train)
        X_test = np.matrix(X_test)
        Y_train = np.matrix(Y_train).getT()
        Y_test = np.matrix(Y_test).getT()

        i_testsize = 0
        i_test = 0
        i_train = 0
        
        for i in range(self.m):
            if(i_testsize< self.testsize and i == self.testindex[i_testsize]):
                
#                print "test", i
                for j in range(self.n+1):
                    X_test[i_test, j] = self.X[i,j]
                Y_test[i_test, 0] = self.Y[i,0]
                i_test = i_test + 1
                i_testsize = i_testsize+1
            else:
 #               print "train", i
                for j in range(self.n+1):
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
        X = np.ones((self.m,self.n+1))
        Y = np.ones((self.m))
        for line in lines:
            s = " ".join(line.split())
            s = s.split()
            s = [ float(x) for x in s ]
            for j in range(len(s)-1):
                X[i,j+1] = s[j]
            Y[i] = s[len(s)-1]
            i = i+1
        return  np.matrix(X), np.matrix(Y).getT()
        
    def get_Grad(self):
        a = ((self.X_train*self.Beta - self.Y_train).getT())*self.X_train
        return (a*self.lr).getT()
    
    def get_sqerr(self,a,b):
        return (np.sum( np.square(a -b), axis = 0))[0,0]/(len(a))

    def test(self, X, Y, Beta):
        return self.get_sqerr(X*self.Beta, Y)

    def train(self,Beta,X,Y,lr):
        i =0
        meansqerror = self.get_sqerr(X*Beta,Y)
        preverror = meansqerror
        trainErr = []
        validErr = []
        while (i<1000000):
            a = self.get_Grad()
            self.Beta = self.Beta - a
            meansqerror = self.get_sqerr(self.X*self.Beta, self.Y)
            if i%100 == 0:
                print meansqerror
            i = i+1
            trainErr.append(meansqerror)
            validErr.append(self.test(self.X_test, self.Y_test, self.Beta))
            if(meansqerror >= preverror or (meansqerror < preverror and abs(meansqerror - preverror) < 0.00000001)):
                print meansqerror, preverror
                break
            else:
                preverror = meansqerror
        return Beta,trainErr,validErr,i

f = sys.argv[1]
features = 13
model = LinearRegression(f,features, 0.2)


beta, trainErr, validErr, n_epoch = model.train(model.Beta,model.X_train, model.Y_train ,model.lr)

testErr = model.test(model.X_test, model.Y_test, model.Beta) 
print "Test error ", testErr
print "Train error ",  model.test(model.X_train, model.Y_train, model.Beta)

def plot_learning_curve(train_Err,validErr, n_epoch):
    print "length of train_Err ", len(train_Err), "n_epoch ", n_epoch
    plt.plot(np.arange(n_epoch), validErr, np.arange(n_epoch), train_Err,  'k--')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend(('Validation_Err', 'Train Error'), shadow=True, loc=(0.01, 0.55))
    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize=20, color='b')
    plt.setp(ltext[1], fontsize=20, color='g')
    plt.show()
    
plot_learning_curve(trainErr,validErr,n_epoch)
