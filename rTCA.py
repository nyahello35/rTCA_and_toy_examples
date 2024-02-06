import numpy as np
import math
from numpy.linalg import eig, norm, inv


def rTCA(training_input, testing_input, mu, sigma, kernel, dim, propotion , r):
    n_training_input = training_input.shape[0]
    n_testing_input = testing_input.shape[0]
    n_total = n_training_input + n_testing_input

    total_input = np.vstack((training_input,testing_input))
    # Randomly select the subset
    rng = np.random.default_rng(r)
    nSubset = int(n_total*propotion)
    index = rng.permutation(np.arange(n_total))[0:nSubset]
    sub_input = total_input[index]
    # Construct kernel matrix
    K = np.zeros((n_total,nSubset))
    match kernel:
        case 'linear':
            for i in range(n_total):
                for j in range(nSubset):
                    K[i,j] = np.inner(total_input[i],sub_input[j])
        case 'rbf':
            for i in range(n_total):
                for j in range(nSubset):
                    K[i,j] = math.exp(-2*math.pow(norm(total_input[i]-sub_input[j]),2)/(2*sigma))
    # Construct other matrix
    H = np.eye(n_total) - (1/n_total)*np.ones((n_total,n_total)) #centering matrix
    L_sub1 = (1/n_training_input**2)*np.ones((n_training_input,n_training_input))
    L_sub2 = (-1/(n_training_input*n_testing_input))*np.ones((n_training_input,n_testing_input))
    L_sub3 = (1/n_testing_input**2)*np.ones((n_testing_input,n_testing_input))
    L = np.vstack((np.hstack((L_sub1,L_sub2)),np.hstack((np.transpose(L_sub2),L_sub3))))

    # Compute transpose component
    target = inv(np.transpose(K)@L@K+mu*np.eye(r))@np.transpose(K)@H@K
    eigenvalue, eigenvector = eig(target)
    d = np.argsort(eigenvalue)[0:dim]
    W = np.transpose(np.transpose(eigenvector)[d])
    #print(W.shape)
    total_input_tca = np.real(K@W)
    training_input_tca = total_input_tca[0:n_training_input]
    testing_input_tca = total_input_tca[n_training_input:]
    return training_input_tca, testing_input_tca