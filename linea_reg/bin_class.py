import numpy as np
import sys
import matplotlib.pyplot as plt
import math
train = int(sys.argv[3])
class LogisticRegression(object):

    def __init__(self, trainfileName, testfileName):
        self.alpha = 0.000001
        self.features = 0
        self.read_data(trainfileName, testfileName)
        self.Beta = np.random.normal(0,0.01,(self.features+1,1))
        self.threshold = 0.0
        self.iter = 100000000
        self.mini = 20
        self.l2 = 0.001
    def read_data(self, testfile, trainfile):
        testfile = open(testfile, 'r')
        trainfile = open(trainfile, 'r')
        lines_train  = trainfile.readlines()
        lines_test = testfile.readlines()
        testfile.close()
        trainfile.close()
        
        self.trainsize = len(lines_train)
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
                self.Y_train = np.ones((self.trainsize,1))
                len_s = len(s)
            for j in range(1,len_s):
                self.X_train[i, j] = s[j]
            self.Y_train[i,0] = s[0]
            i = i+1

        #self.X_train = np.asarray(self.X_train, dtype = int)
        #self.Y_train = np.asarray(self.Y_train, dtype = int)
        
        i = 0
        for line in lines_test:
            s = " ".join(line.split(","))
            s = s.split()
            s = [int(x) for x in s]
            if i == 0:
                self.X_test = np.ones((self.testsize, self.features+1))
                self.Y_test = np.ones((self.testsize,1))
                len_s = len(s)
            for j in range(1,len_s):
                self.X_test[i,j] = s[j]
            self.Y_test[i,0] = s[0]
            i = i+1
    def get_cost(self, a, b, size):
        c = np.dot(a,self.Beta)
        c = np.log(1/(1+np.exp(-c)))
        d = np.log(1-c)
        b1 = 1- b
        #print np.sum(((b*c) + (b1*d))*(-1)),np.sum(self.Beta*self.Beta*self.l2/2)
        return (np.sum(((b*c) + (b1*d))*(-1))+np.sum(self.Beta*self.Beta*self.l2/2))/size

    def get_Grad(self, X, Y):
        a = np.dot(X,self.Beta)
        a = 1/(1+np.exp(-a))
        c = np.repeat((a-Y),(self.features+1)).shape
        a = np.repeat((a-Y),(self.features+1)).reshape(len(a),(self.features+1))
        grad = a*X
        
        #print grad.shape
        #print (np.mean(grad*self.lr,axis=0).reshape(self.features+1,1)+self.l2*self.Beta).shape
        return (np.mean(grad*self.lr,axis=0)).reshape(self.features+1,1)+self.l2*self.Beta
        
    def train(self, X, Y, size):
        costList = []
        cost = self.get_cost(X,Y,size)
        print "initial cost ", cost
        prevcost = cost
        self.lr = self.alpha
        i = 0
        n_batch = self.trainsize/self.mini
        while(i < self.iter):
            batch_index = i%n_batch
            rand_ind = np.random.permutation(self.trainsize)[0:self.mini]
            X_ind = X[rand_ind]
            Y_ind = Y[rand_ind]
            grad = self.get_Grad(X_ind,Y_ind)
            #print "Updating"
            #print self.Beta.shape,grad.shape
            self.Beta = self.Beta - grad
            #print self.Beta.shape
            #print 
            cost = self.get_cost(X_ind,Y_ind,size)
            costList.append(cost)
            if i%100000 == 0:
                print cost
            i = i+1

            #if(cost >= prevcost or (cost < prevcost and abs(cost - prevcost) < 0.0000001)):
                #print cost, prevcost
                #break
            #else:
            #    prevcost = cost
        return self.Beta, costList, i

    def test(self, X, Y, size, Beta):
        a = np.dot(X,Beta)
        a = 1/(1+np.exp(-a))
        a[a<self.threshold]=0
        a[a>=self.threshold]=1
        return self.get_accuracy(a, Y)
    

    def get_accuracy(self, a,b):
        count = 0
        for i in range(a.size):
            if(a[i] == b[i]):
                count = count + 1
        return ((float(count))/a.size)*100
    
trainf = sys.argv[1]
testf = sys.argv[2]

model = LogisticRegression(trainf, testf)
if train == 1:
    Beta, costList, n_epoch = model.train(model.X_train, model.Y_train, model.trainsize)

    f = open("ares.txt",'w')
    for i in range(model.features+1):
        f.write("%s\n" % str(Beta[i,0]))
    f.close()
else:
    Beta = np.ones((model.features+1,1))
    f = open("ares.txt",'r')
    lines = f.readlines()
    lines= [float(x.strip('\n')) for x in lines]
    Beta = np.asarray(lines)
max_accuracy  = 0
best_thresh = 0
model.threshold = 0.0
start_thresh = 0
update_thresh = 0.001
a=0.51
b = 0.53
model.threshold = a
for i in range(0,int((b-a)/update_thresh)):
    accuracy  = model.test(model.X_train, model.Y_train, model.trainsize, Beta)
    model.threshold += update_thresh
    #print accuracy,model.threshold
    if accuracy>max_accuracy:
        start_thresh = model.threshold
    if accuracy >=max_accuracy:
        max_accuracy = accuracy
        best_thresh = model.threshold
print max_accuracy,start_thresh,best_thresh
model.threshold = best_thresh
accuracy  = model.test(model.X_test, model.Y_test, model.testsize, Beta)
print "Test accuracy"
print accuracy
def plot_learning_curve(costList, n_epoch):
    print "length of costList ", len(costList), " n_epoch ", n_epoch
    plt.plot(np.arange(n_epoch), np.array(costList))
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.show()

#plot_learning_curve(costList, n_epoch)
