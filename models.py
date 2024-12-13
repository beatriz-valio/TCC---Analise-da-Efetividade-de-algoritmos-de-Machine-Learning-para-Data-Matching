from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import utils
import joblib

'''
Avalia os algoritmos
'''

def avaliate_algoritms(data_formatada, dataset_name, study_rf, study_svm, study_nn, X_train, y_train, X_test, y_test):
    
    # Avaliação do Random Forest
    timestamp1 = datetime.now()
    rf_best_model = RandomForestClassifier(**study_rf.best_params, random_state=42)
    rf_best_model.fit(X_train, y_train)

    # Previsões no conjunto de teste
    rf_predictions = rf_best_model.predict(X_test)
    rf_accuracy, rf_precision, rf_recall, rf_f1 = utils.process_metrics(rf_best_model, X_train, y_train)
    rf_accuracy_prod, rf_precision_prod, rf_recall_prod, rf_f1_prod = utils.process_metrics(rf_best_model, X_test, y_test)

    cm_rf = utils.plot_confusion_matrix(rf_best_model, 'Random Forest', X_train, y_train, f'confusion_matrix_rf_{data_formatada}_{dataset_name}.png')
    cm_rf_prod = utils.plot_confusion_matrix(rf_best_model, 'Random Forest', X_test, y_test, f'confusion_matrix_rf_prod_{data_formatada}_{dataset_name}.png')
    timestamp_rf = datetime.now() - timestamp1
    
    print("\n\nRF - Métricas melhor hiperparâmetro")
    print(f"Data: {data_formatada}, Tempo de Execução: {timestamp_rf}")
    print(f"Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}, F1-Score: {rf_f1}")

    # Avaliação do SVM
    timestamp2 = datetime.now()
    svm_best_model = SVC(**study_svm.best_params, random_state=42)
    svm_best_model.fit(X_train, y_train)

    # Previsões no conjunto de teste
    svm_predictions = svm_best_model.predict(X_test)
    svm_accuracy, svm_precision, svm_recall, svm_f1 = utils.process_metrics(svm_best_model, X_train, y_train)
    svm_accuracy_prod, svm_precision_prod, svm_recall_prod, svm_f1_prod = utils.process_metrics(svm_best_model, X_test, y_test)

    cm_svm = utils.plot_confusion_matrix(svm_best_model, 'SVM', X_train, y_train, f'confusion_matrix_svm_{data_formatada}_{dataset_name}.png')
    cm_svm_prod = utils.plot_confusion_matrix(svm_best_model, 'SVM', X_test, y_test, f'confusion_matrix_svm_prod_{data_formatada}_{dataset_name}.png')
    timestamp_svm = datetime.now() - timestamp2
    
    print("\n\nSVM - Métricas melhor hiperparâmetro")
    print(f"Data: {data_formatada}, Tempo de Execução: {timestamp_svm}")
    print(f"Accuracy: {svm_accuracy}, Precision: {svm_precision}, Recall: {svm_recall}, F1-Score: {svm_f1}")

    # Avaliação da Rede Neural
    best_params = study_nn.best_params
    n_layers = best_params.pop('n_layers')
    hidden_layer_sizes = tuple(best_params.pop(f"hidden_layer_{i}_size") for i in range(n_layers))
    best_params['hidden_layer_sizes'] = hidden_layer_sizes
    
    timestamp3 = datetime.now()
    nn_best_model = MLPClassifier(**best_params, random_state=42)
    nn_best_model.fit(X_train, y_train)

    # Previsões no conjunto de teste
    nn_predictions = nn_best_model.predict(X_test)
    nn_accuracy, nn_precision, nn_recall, nn_f1 = utils.process_metrics(nn_best_model, X_train, y_train)
    nn_accuracy_prod, nn_precision_prod, nn_recall_prod, nn_f1_prod = utils.process_metrics(nn_best_model, X_test, y_test)

    cm_nn = utils.plot_confusion_matrix(nn_best_model, 'Neural Network', X_train, y_train, f'confusion_matrix_nn_{data_formatada}_{dataset_name}.png')
    cm_nn_prod = utils.plot_confusion_matrix(nn_best_model, 'Neural Network', X_test, y_test, f'confusion_matrix_nn_prod_{data_formatada}_{dataset_name}.png')
    timestamp_nn = datetime.now() - timestamp3
    
    print("\n\nNN - Métricas melhor hiperparâmetro")
    print(f"Data: {data_formatada}, Tempo de Execução: {timestamp_nn}")
    print(f"Accuracy: {nn_accuracy}, Precision: {nn_precision}, Recall: {nn_recall}, F1-Score: {nn_f1}")

    '''
    Processa métricas obtidas
    '''
    # Criar DataFrame para armazenar métricas e datas
    metrics_df = pd.DataFrame(columns=['Data', 'Tipo', 'Dataset', 'Modelo', 'Acurácia', 'Precisão', 'Revocação', 'F1-Score', 'TP', 'TN', 'FP', 'FN', 'Melhor Hiperparâmetro', 'Tempo de Execução'])

    # Adicionar métricas do SVM
    new_data = pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Treino', 'Dataset':[dataset_name], 'Modelo': ['SVM'], 
                            'Acurácia': [svm_accuracy], 'Precisão': [svm_precision], 
                            'Revocação': [svm_recall], 'F1-Score': [svm_f1], 
                            'TP': [cm_svm[1, 1]], 'TN': [cm_svm[0, 0]], 
                            'FP': [cm_svm[0, 1]], 'FN': [cm_svm[1, 0]], 
                            'Melhor Hiperparâmetro': str(study_svm.best_params),
                            'Tempo de Execução': timestamp_svm})
    metrics_df = pd.concat([metrics_df, new_data], ignore_index=True)

    # Adicionar métricas do Random Forest
    metrics_df = pd.concat([metrics_df, pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Treino', 'Dataset':[dataset_name], 'Modelo': ['RF'], 
                                                      'Acurácia': [rf_accuracy], 'Precisão': [rf_precision],
                                                      'Revocação': [rf_recall], 'F1-Score': [rf_f1], 
                                                      'TP': [cm_rf[1, 1]], 'TN': [cm_rf[0, 0]], 
                                                      'FP': [cm_rf[0, 1]], 'FN': [cm_rf[1, 0]], 
                                                      'Melhor Hiperparâmetro': str(study_rf.best_params), 
                                                      'Tempo de Execução': timestamp_rf})], ignore_index=True)

    # Adicionar métricas da Rede Neural
    metrics_df = pd.concat([metrics_df, pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Treino', 'Dataset':[dataset_name], 'Modelo': ['NN'], 
                                                      'Acurácia': [nn_accuracy], 'Precisão': [nn_precision],
                                                      'Revocação': [nn_recall], 'F1-Score': [nn_f1], 
                                                      'TP': [cm_nn[1, 1]], 'TN': [cm_nn[0, 0]], 
                                                      'FP': [cm_nn[0, 1]], 'FN': [cm_nn[1, 0]], 
                                                      'Melhor Hiperparâmetro': str(study_nn.best_params), 
                                                      'Tempo de Execução': timestamp_nn})], ignore_index=True)

    # Verificar DataFrame resultante
    print("\n\nMetrics DataFrame:")
    print(metrics_df)

    # Plota as métricas
    utils.plot_metrics({
        'Acurácia': metrics_df['Acurácia'].tolist(),
        'Precisão': metrics_df['Precisão'].tolist(),
        'Revocação': metrics_df['Revocação'].tolist(),
        'F1-Score': metrics_df['F1-Score'].tolist(),
    }, f'models_metrics_{data_formatada}_{dataset_name}.png')


    '''
    Processa métricas simulando PROD
    '''
    # Criar DataFrame para armazenar métricas e datas
    metrics_df_prod = pd.DataFrame(columns=['Data', 'Tipo', 'Dataset', 'Modelo', 'Acurácia', 'Precisão', 'Revocação', 'F1-Score', 'TP', 'TN', 'FP', 'FN', 'Melhor Hiperparâmetro', 'Tempo de Execução'])

    # Adicionar métricas do SVM
    new_data = pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Teste', 'Dataset':[dataset_name], 'Modelo': ['SVM'], 
                             'Acurácia': [svm_accuracy_prod], 'Precisão': [svm_precision_prod], 
                            'Revocação': [svm_recall_prod], 'F1-Score': [svm_f1_prod], 
                            'TP': [cm_svm_prod[1, 1]], 'TN': [cm_svm_prod[0, 0]], 
                            'FP': [cm_svm_prod[0, 1]], 'FN': [cm_svm_prod[1, 0]], 
                            'Melhor Hiperparâmetro': str(study_svm.best_params),  
                            'Tempo de Execução': timestamp_svm})
    metrics_df_prod = pd.concat([metrics_df_prod, new_data], ignore_index=True)

    # Adicionar métricas do Random Forest
    metrics_df_prod = pd.concat([metrics_df_prod, pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Teste', 'Dataset':[dataset_name], 'Modelo': ['RF'], 
                                                      'Acurácia': [rf_accuracy_prod], 'Precisão': [rf_precision_prod],
                                                      'Revocação': [rf_recall_prod], 'F1-Score': [rf_f1_prod], 
                                                      'TP': [cm_rf_prod[1, 1]], 'TN': [cm_rf_prod[0, 0]], 
                                                      'FP': [cm_rf_prod[0, 1]], 'FN': [cm_rf_prod[1, 0]], 
                                                      'Melhor Hiperparâmetro': str(study_rf.best_params), 
                                                      'Tempo de Execução': timestamp_rf})], ignore_index=True)

    # Adicionar métricas da Rede Neural
    metrics_df_prod = pd.concat([metrics_df_prod, pd.DataFrame({'Data': [data_formatada], 'Tipo': 'Teste', 'Dataset':[dataset_name], 'Modelo': ['NN'], 
                                                      'Acurácia': [nn_accuracy_prod], 'Precisão': [nn_precision_prod],
                                                      'Revocação': [nn_recall_prod], 'F1-Score': [nn_f1_prod], 
                                                      'TP': [cm_nn_prod[1, 1]], 'TN': [cm_nn_prod[0, 0]], 
                                                      'FP': [cm_nn_prod[0, 1]], 'FN': [cm_nn_prod[1, 0]], 
                                                      'Melhor Hiperparâmetro': str(study_nn.best_params), 
                                                      'Tempo de Execução': timestamp_nn})], ignore_index=True)

    # Verificar DataFrame resultante
    print("\n\nMetrics DataFrame:")
    print(metrics_df_prod)

    # Plota as métricas
    utils.plot_metrics({
        'Acurácia': metrics_df_prod['Acurácia'].tolist(),
        'Precisão': metrics_df_prod['Precisão'].tolist(),
        'Revocação': metrics_df_prod['Revocação'].tolist(),
        'F1-Score': metrics_df_prod['F1-Score'].tolist(),
    }, f'models_metrics_prod_{data_formatada}_{dataset_name}.png')


    '''
    Salvar modelos
    '''
    joblib.dump(rf_best_model, f'./best_models/rf_best_model_{data_formatada}_{dataset_name}.pkl')
    joblib.dump(svm_best_model, f'./best_models/svm_best_model_{data_formatada}_{dataset_name}.pkl')
    joblib.dump(nn_best_model, f'./best_models/nn_best_model_{data_formatada}_{dataset_name}.pkl')

    return metrics_df, metrics_df_prod, rf_best_model, svm_best_model, nn_best_model
