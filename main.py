from rTCA import rTCA
from data_generator import data_generator
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# Set the experiment parameter
np.random.seed(1)
mu = np.array([1,0,0])
A = 2*np.random.random((3,3))-1
var = np.transpose(A)@A
class_shift = np.array([3,0,0])
translation = np.array([6,6,6])
angle = 0
nsp = 100
nsn = 100
ntp = 100
ntn = 100
plot = 0
# Parameter for rTCA
mu_rtca = 0.1
sigma = 1
kernel = 'linear'
dim = 2
proportion = 0.1
r = 1

# Generate 3-dimensional data with datasets shift
Xsp, Xsn, Xtp, Xtn = data_generator(mu, var, class_shift, translation, angle, nsp, nsn, ntp, ntn)
Ysp = np.array([1]*nsp)
Ysn = np.array([0]*nsn)
Ytp = np.array([1]*ntp)
Ytn = np.array([0]*ntn)
if plot == 1:
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    for data, m, c in [(Xsp,'o','r'),(Xsn,'x','r'),(Xtp,'o','b'),(Xtn,'x','b')]:
        ax.scatter(np.transpose(data)[0],np.transpose(data)[1],np.transpose(data)[2], marker = m , c = c)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
plt.show()

# Use svm to classify the original target dataset
Xs = np.vstack((Xsp,Xsn))
Ys = np.hstack((Ysp,Ysn))
Xt = np.vstack((Xtp,Xtn))
Yt = np.hstack((Ytp,Ytn))
mdl1 = SVC(gamma='auto')
mdl1.fit(Xs,Ys)

Yt_pred = mdl1.predict(Xt)
acc1 = np.sum(Yt_pred == Yt) / (ntp+ntn)
print(acc1)

# rTCA
acc2 = []
Xs_rtca, Xt_rtca = rTCA(Xs,Xt,mu,sigma,kernel,dim,proportion, r)
Xsp_rtca = Xs_rtca[0:nsp]
Xsn_rtca = Xs_rtca[nsp+1:]
Xtp_rtca = Xt_rtca[0:ntp]
Xtn_rtca = Xt_rtca[ntp+1:]
mdl2 = SVC(gamma='auto')
mdl2.fit(Xs_rtca,Ys)

Yt_pred_rtca = mdl2.predict(Xt_rtca)
acc2 = (np.array([np.sum(Yt_pred_rtca == Yt) / (ntp+ntn)]))



if plot == 1:
    for data, m, c in [(Xsp_rtca,'o','r'),(Xsn_rtca,'x','r'),(Xtp_rtca,'o','b'),(Xtn_rtca,'x','b')]:
        plt.scatter(np.transpose(data)[0],np.transpose(data)[1], marker = m , c = c)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


#print(acc2)