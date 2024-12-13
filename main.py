from datetime import datetime
from sklearn.model_selection import train_test_split
from colorama import Fore, init
import utils, feature_selection, transformations, hyperparameter_optimization, models

init(autoreset=True)

print(Fore.MAGENTA + "\n ------ Iniciando execução ------")

data_inicial = datetime.now()
data_formatada = data_inicial.strftime("%d_%m_%Y_%H_%M_%S")
print(Fore.BLUE + f"{data_inicial}\n")

'''
Iniciando dataset
'''
print(Fore.MAGENTA + "\n ------ Carregando dados ------")

# Carregar os dados do CSV
dataset_csv = "padrao_ouro_notas_fiscais.csv"
delimiter = ";"
target_column = "matching"
X, y = utils.load_dataset(dataset_csv, delimiter, target_column)

'''
Seleção e transformação de features
'''

print(Fore.MAGENTA + "\n ------ Iniciando seleção e transformação ------")

# Preprocessar colunas de datas
date_columns = ['data_empenho', 'data_assinatura', 'data_liquidacao', 'data_vencimento', 'data_emissao', 'data_saida']
X_full = utils.preprocess_dates(X, date_columns)

# Identificar atributos: definindo o tipo das variaveis para devida transformacao
numeric_columns = ['valor_liquidacao', 'valor_total_itens'] + [col + '_mes' for col in date_columns] + [col + '_dia' for col in date_columns] 
categoric_columns = []
text_columns = ['nro_contrato', 'numero_contrato', 'credor', 'nome_contratado', 
                    'nome_emitente', 'nome_fant_emitente', 'cnpj_cpf_credor', 'codigo_cic_contratado', 
                    'cnpj_emitente', 'descricao_objeto_licitacao', 'descricao_objetivo', 'descricao_produto']
key_columns = []

# Remove do dataset features que não são dados relevantes no matching, mas apenas identificadores das tabelas originárias
X_full = X_full.drop(columns=key_columns)


# Selecionar automaticamente features
print(Fore.MAGENTA + "\n ------ Iniciando seleção ------")
X_selected, features_index_selected = feature_selection.select_all_type_of_features(X_full, y, data_formatada, categoric_columns, text_columns)

# Trasformar automaticamente features
print(Fore.MAGENTA + "\n ------ Iniciando transformação ------")
X_combined_transformed = transformations.apply_transformation_all_features(X_full, features_index_selected, numeric_columns, categoric_columns, text_columns)

# Dividir treino e teste para cada um (20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X_combined_transformed, y, test_size=0.2, random_state=42)

# Otimizar hiperparametros
print(Fore.MAGENTA + "\n ------ Iniciando otimização de hiperparâmetros ------")
study_rf, best_params_rf_str, study_svm, best_params_svm_str, study_nn, best_params_nn_str = hyperparameter_optimization.optimize(X_train, y_train)

# Avaliar cada algoritmo
print(Fore.MAGENTA + "\n ------ Iniciando avaliação dos modelos para testes ------")
dataset_name = dataset_csv.split('.')[0]
metrics_df, metrics_df_prod, rf_best_model, svm_best_model, nn_best_model = models.avaliate_algoritms(data_formatada, dataset_name, study_rf, study_svm, study_nn, X_train, y_train, X_test, y_test)

utils.save_metrics(metrics_df)
utils.save_metrics(metrics_df_prod)

'''
Finalizando e organizando documentos gerados
'''
utils.organize_docs(data_formatada, dataset_name)

print(Fore.GREEN + "\n ------ Script finalizado ------")

data_termino = datetime.now()
print('Tempo total de execução:', data_termino - data_inicial)
