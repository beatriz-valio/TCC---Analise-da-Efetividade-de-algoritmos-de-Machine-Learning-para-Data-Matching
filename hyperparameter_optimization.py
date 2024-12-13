from colorama import Fore
import optuna
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

n_trials = 5

# Cross-validation: 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

'''
Aqui a ideia já é buscar automaticamente os melhores hiperparametros para cada
um dos algoritmos. Estou utilizando precisão pra essa definição.
'''

'''
Busca por melhores hiperparâmetros
'''

# Função de objetivo para Random Forest com validação cruzada
def objective_rf(trial, X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators', 10, 200),
        max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        max_depth=trial.suggest_int('max_depth', 5, 50),
        criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
        random_state=42
    )
    scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='precision') # Para minimizar falsos positivos
    return np.mean(scores)

# Função de objetivo para SVM com validação cruzada
def objective_svm(trial, X_train, y_train):
    svm_model = SVC(
        C=trial.suggest_float('C', 1e-6, 1e2, log=True),
        gamma=trial.suggest_float('gamma', 1e-6, 1e2, log=True),
        kernel=trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        probability=True,
        random_state=42
    )
    scores = cross_val_score(svm_model, X_train, y_train, cv=kf, scoring='precision') # Para minimizar falsos positivos
    return np.mean(scores)

# Função de objetivo para Rede Neural (MLPClassifier) com validação cruzada
def objective_nn(trial, X_train, y_train):
    hidden_layers = tuple(trial.suggest_int(f"hidden_layer_{i}_size", 10, 200) for i in range(trial.suggest_int("n_layers", 1, 3)))
    nn_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
        solver=trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
        alpha=trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
        learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
        max_iter=3000,
        random_state=42
    )
    scores = cross_val_score(nn_model, X_train, y_train, cv=kf, scoring='precision') # Para minimizar falsos positivos
    return np.mean(scores)

# Otimização de hiperparametros
def optimize(X_train, y_train):
    print(Fore.BLUE + "\nIniciando estudo de hiperparâmetros para RF.")
    study_rf = optuna.create_study(direction='maximize', study_name="RF")
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=n_trials)
    print(Fore.GREEN + f"\nSucesso! Melhores hiperparâmetros RF: {study_rf.best_params}")
    best_params_rf_str = str(study_rf.best_params)

    print(Fore.BLUE + "\nIniciando estudo de hiperparâmetros para SVM.")
    study_svm = optuna.create_study(direction='maximize', study_name="SVM")
    study_svm.optimize(lambda trial: objective_svm(trial, X_train, y_train), n_trials=n_trials)
    print(Fore.GREEN + f"\nSucesso! Melhores hiperparâmetros SVM: {study_svm.best_params}")
    best_params_svm_str = str(study_svm.best_params)

    print(Fore.BLUE + "\nIniciando estudo de hiperparâmetros para NN.")
    study_nn = optuna.create_study(direction='maximize', study_name="NN")
    study_nn.optimize(lambda trial: objective_nn(trial, X_train, y_train), n_trials=n_trials)
    print(Fore.GREEN + f"\nSucesso! Melhores hiperparâmetros NN: {study_nn.best_params}")
    best_params_nn_str = str(study_nn.best_params)

    return study_rf, best_params_rf_str, study_svm, best_params_svm_str, study_nn, best_params_nn_str
