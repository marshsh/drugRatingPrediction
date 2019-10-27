from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score

#####################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import layers
from keras.regularizers import l1, l2

#####################################################






def show_metrics(title, y_true, y_regr, y_clas):
    y_regr_closest = np.round(y_regr)
    
    fmt = '{:<16} | {:>8} | {:>8} | {:>8}'.format
    nums2str = lambda *nums: (f'{n:.3f}' for n in nums)
    
    print(fmt(title, 'MAE', 'KAPPA', 'ACCURACY'))
    print(fmt(' regression', *nums2str( 
            mean_absolute_error(y_true, y_regr),
            cohen_kappa_score(y_true, y_regr_closest),
            accuracy_score(y_true, y_regr_closest)
    )))
    
    print(fmt(' classification', *nums2str( 
            mean_absolute_error(y_true, y_clas),
            cohen_kappa_score(y_true, y_clas),
            accuracy_score(y_true, y_clas)
    )))
    print()
