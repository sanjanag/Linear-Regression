import numpy as np
import sys
import matplotlib.pyplot as plt

class LinearRegression(object):
    
    def __init__(self,fileName,n):
        self.n = n
        self.m = 0
        self.X, self.Y= self.read_data(fileName)
        self.alpha = 0.000006
        self.Beta = np.matrix( np.zeros((n+1))).getT()
        self.lr = np.identity(n+1)
        for i in range(n+1):
            self.lr[i,i] = self.alpha/self.m
            
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
        a = ((self.X*self.Beta - self.Y).getT())*self.X
        return (a*self.lr).getT()
    
    def get_sqerr(self,a,b):
        return (np.sum( np.square(a -b), axis = 0))[0,0]/(2*self.m)
    

    def train(self,Beta,X,Y,lr):
        i =0
        meansqerror = self.get_sqerr(X*Beta,Y)
        preverror = meansqerror
        trainErr = []
        while (1):
            a = self.get_Grad()
            self.Beta = self.Beta - a
            meansqerror = self.get_sqerr(self.X*self.Beta, self.Y)
            if i%100 == 0:
                print meansqerror
            i = i+1
            trainErr.append(meansqerror)
            if(meansqerror >= preverror or (meansqerror < preverror and abs(meansqerror - preverror) < 0.00000001)):
                print meansqerror, preverror
                break
            else:
                preverror = meansqerror
        return Beta,trainErr,i

f = sys.argv[1]
features = 13
model = LinearRegression(f,features)

beta, trainErr, n_epoch = model.train(model.Beta,model.X,model.Y,model.lr)

def plot_learning_curve(train_Err,n_epoch):
    print len(train_Err),n_epoch
    plt.plot(np.arange(n_epoch), train_Err, 'b-')
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()
plot_learning_curve(trainErr,n_epoch)
