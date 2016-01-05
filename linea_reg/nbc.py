import numpy as np
import sys
import matplotlib.pyplot as plt
import math

class NaiveBayesClassifier(object):

    def __init__(self, trainfileName, testfileName):
        self.features = 0
        self.read_data(trainfileName, testfileName)
        self.classes = np.unique(self.Y_train)
#        print self.classes.size

    def estimate(self,X,Y):
        freq = np.zeros((self.classes.size))
        for i in range(self.classes.size):
            for j in range(self.trainsize):
                if Y[j]== i:
                    freq[i]+=1

        self.prob_class = freq/self.trainsize
        freq_f = np.zeros((self.features))

        for i in range(self.features):
            for j in range(self.trainsize):
                if X[j,i]==1:
                    freq_f[i]+=1

        self.prob_feature = freq_f/self.trainsize

        freq_class_feature = np.zeros((self.features, self.classes.size))

        for i in range(self.features):
            for k in range(self.trainsize):
                if X[k,i]==1 :
                    freq_class_feature[i,Y[k]] +=1

        freq_tile = np.tile(freq,(self.features,1))

        self.prob_feature_class = freq_class_feature/freq_tile

    def test(self,X):
        Y = np.zeros((self.testsize))
        for i in range(self.testsize):
            prob = np.ones((self.classes.size))
            prior = 1
            evidence = 1
            for j in range(self.classes.size):
                prob[j] = prob[j]*self.prob_class[j]
                for k in range(self.features):
                    if X[i,k]==1:
                        prob[j] = prob[j]*self.prob_feature_class[k,j]
                    else:
                        prob[j] = prob[j]*(1-self.prob_feature_class[k,j])
            for j in range(self.features):
                if X[i,j]==1:
                    evidence = evidence*self.prob_feature[j]
                else:
                    evidence = evidence*(1-self.prob_feature[j])
            prob = prob/evidence
            Y[i] = np.argmax(prob)
        return Y

    def get_accuracy(self, a,b):
        count = 0
        for i in range(a.size):
            if(a[i] == b[i]):
                count = count + 1
        return ((float(count))/a.size)*100
    


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
                self.X_train = np.ones((self.trainsize, self.features),dtype = int)
                self.Y_train = np.ones((self.trainsize,1),dtype = int)
                len_s = len(s)
            for j in range(1,len_s):
                self.X_train[i, j-1] = s[j]
                self.Y_train[i,0] = s[0]
            i = i+1

        
        i = 0
        for line in lines_test:
            s = " ".join(line.split(","))
            s = s.split()
            s = [int(x) for x in s]
            if i == 0:
                self.X_test = np.ones((self.testsize, self.features))
                self.Y_test = np.ones((self.testsize,1))
                len_s = len(s)
            for j in range(1,len_s):
                self.X_test[i,j-1] = s[j]
                self.Y_test[i,0] = s[0]
            i = i+1



trainf = sys.argv[1]
testf = sys.argv[2]

model = NaiveBayesClassifier(trainf,testf)

model.estimate(model.X_train,model.Y_train)

guess=model.test(model.X_test)
print guess
accuracy = model.get_accuracy(model.Y_test,guess)
print accuracy
