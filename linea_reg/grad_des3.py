import numpy as np
import sys
import matplotlib.pyplot as plt
import math
#from sklearn.cross_validation import train_test_split
import random

class LinearRegression(object):
    
    def __init__(self,fileName, n, ratio):
        
        self.n = n
        self.m = 0
        self.ratio = ratio
            
    
        self.X, self.Y= self.read_data(fileName)
        
        self.mini = 150
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.split(self.X, self.Y)
       # print self.Y_test
        self.alpha = 0.0000007
        self.Beta = np.matrix( np.zeros((n+1))).getT()
        self.lr = np.identity(n+1)
        for i in range(n+1):
            self.lr[i,i] = self.alpha/self.mini

    def split(self, X, y):
        indices = np.random.permutation(self.m)
        self.testsize = int(self.ratio * self.m)
        self.testindex = indices[0:self.testsize]
        self.trainindex = indices[self.testsize+1:self.m]
        return X[self.trainindex],X[self.testindex],y[self.trainindex],y[self.testindex]
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
        
    def get_Grad(self,X,y):
        a = ((X*self.Beta - y).getT())*X
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
        minibatch = 0
        miniBatchNumber = math.ceil(len(self.Y_train)/self.mini*1.0)
        while (i<1000000):
            minibatch = 0
            while minibatch<miniBatchNumber:
                miniBatchindex = np.arange(minibatch*self.mini,min(minibatch*self.mini+self.mini,self.m))
                minibatch = minibatch + 1
                mX = self.X_train[miniBatchindex]
                mY = self.Y_train[miniBatchindex]
                a = self.get_Grad(mX,mY)
                self.Beta = self.Beta - a
            meansqerror = self.get_sqerr(self.X_train*self.Beta,self.Y_train)
            trainErr.append(meansqerror)
            valid_Err = self.test(self.X_test, self.Y_test, self.Beta)
            validErr.append(valid_Err)
            if i%1000 == 0:
                print meansqerror
            if(valid_Err >= 10*preverror or (valid_Err < preverror and abs(valid_Err - preverror) < 0.00000001)):
                break
            else:
                preverror = valid_Err
            i = i+1
            
        return Beta,trainErr,validErr,i

f = sys.argv[1]
features = 13
model = LinearRegression(f,features, 0.2)


beta, trainErr, validErr, n_epoch = model.train(model.Beta,model.X_train, model.Y_train ,model.lr)

testErr = model.test(model.X_test, model.Y_test, model.Beta) 
print "Test error ", testErr
print "Train error ",  model.test(model.X_train, model.Y_train, model.Beta)
#print model.X_test*model.Beta
#print model.Y_test
print model.get_sqerr(model.X*model.Beta,model.Y)
def plot_learning_curve(train_Err,validErr, n_epoch):
    print "length of train_Err ", len(train_Err), "n_epoch ", n_epoch
    plt.plot(np.arange(n_epoch), train_Err, np.arange(n_epoch), validErr,  'k--')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.legend(( 'Train Error','Validation_Err',), shadow=True, loc=(0.01, 0.55))
    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize=20, color='b')
    plt.setp(ltext[1], fontsize=20, color='g')
    plt.show()
    
plot_learning_curve(trainErr,validErr,n_epoch)
