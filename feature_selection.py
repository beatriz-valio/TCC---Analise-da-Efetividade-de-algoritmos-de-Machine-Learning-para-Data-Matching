import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import utils
from catboost import CatBoostClassifier

n_trials = 5

def select_all_type_of_features(df, y, data_formatada, categorical_columns, text_columns):
    # Converter colunas categóricas para o tipo string
    for col in categorical_columns:
        df[col] = df[col].astype(str)

    for col in text_columns:
        df[col] = df[col].astype(str)

    # Imputar valores ausentes
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(df)
    X = pd.DataFrame(X_imputed, columns=df.columns)

    # Função objetivo para CatBoost que valoriza a performance do modelo
    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int('iterations', 100, 500),
            depth=trial.suggest_int('depth', 1, 10),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
            random_seed=42,
            verbose=0  # Silencia a saída de treinamento
        )
        
        # Avaliar a performance do modelo utilizando validação cruzada (5-fold)
        scores = cross_val_score(model, X, y, cv=5, scoring='precision', fit_params={'cat_features': categorical_columns + text_columns})
        return scores.mean()  # Retornar a média da acurácia

    # Executar o Optuna para otimizar a seleção de features com base na performance do modelo
    study = optuna.create_study(direction='maximize', study_name="Feature Selection")
    study.optimize(objective, n_trials=n_trials)

    # Selecionar features com base no melhor modelo encontrado
    best_model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=0)
    best_model.fit(X, y, cat_features=categorical_columns + text_columns)
    
    # Obter a importância das features e filtrar as selecionadas
    feature_importances = best_model.get_feature_importance()
    features_selected = np.where(feature_importances > 0)[0]
    X_selected = X.iloc[:, features_selected]
    
    utils.save_selected_features_to_csv(features_selected, data_formatada, X.columns)  # Salva as features selecionadas

    print("Seleção de características finalizada com sucesso.")

    return X_selected, features_selected
