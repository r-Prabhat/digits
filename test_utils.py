from utils import get_hyperparameter_combinations, split_train_dev_test,read_digits

def test_for_hparam_cominations_count():
    
    gamma_ranges = [0.001, 0.01, 0.1, 1]
    C_ranges = [1, 10, 100, 1000]
    h_params={}
    h_params['gamma'] = gamma_ranges
    h_params['C'] = C_ranges
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    assert len(h_params_combinations) == len(gamma_ranges) * len(C_ranges)

def test_for_hparam_cominations_values():    
    gamma_ranges = [0.001, 0.01]
    C_ranges = [1]
    h_params={}
    h_params['gamma'] = gamma_ranges
    h_params['C'] = C_ranges
    h_params_combinations = get_hyperparameter_combinations(h_params)
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)


def test_data_splitting():
    X, y = read_digits()
    
    X = X[:100,:,:]
    y = y[:100]
    
    test_size = .1
    dev_size = .6

    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert  ((len(X_dev) == 60))