import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    match strategy:
        case "constant":
            if constant is None:
                raise ValueError('invalid constant')
            
            return [constant] * n_test
        
        case 'mean':
            m = np.array(y_train)
            me = m.mean(axis=0)
            return [me] * n_test

        case 'median':
            m = np.array(y_train)
            me = np.median(y_train)
            return [me] * n_test
        
        case 'quantile':
            if quantile is None or not (0 <= quantile <= 1):
                raise ValueError('invalid quantile')
            
            return [np.quantile(y_train, quantile)] * n_test

