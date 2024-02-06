import numpy as np

def data_generator(source_pos_mean, var, class_shift, translation, angle, nsp, nsn, ntp, ntn):
    source_neg_mean = source_pos_mean + class_shift
    Xsp = np.random.multivariate_normal(source_pos_mean, var, nsp)
    Xsn = np.random.multivariate_normal(source_neg_mean, var, nsn)
    rotation_matrix = np.array(([np.cos(angle),-np.sin(angle),0],[np.sin(angle),-np.cos(angle),0],[0,0,1]))
    target_pos_mean = source_pos_mean@rotation_matrix + translation
    target_neg_mean = source_neg_mean@rotation_matrix + translation
    Xtp = np.random.multivariate_normal(target_pos_mean, var, ntp)
    Xtn = np.random.multivariate_normal(target_neg_mean, var, ntn)
    return Xsp, Xsn, Xtp, Xtn