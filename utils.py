import os
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns


'''
Funções de carregamento, salvamento e pre processamento de dados
'''

# Carrega dados separando rótulo de features
def load_dataset(dataset_name, delimeter, target):
    csv_path = 'data/' + dataset_name
    csv = pd.read_csv(csv_path, delimiter=delimeter, encoding="UTF-8")
    y = csv[target]
    x = csv.drop(columns=[target])
    print(f'Dataset {csv_path} carregado com sucesso.')
    return x, y

# Salva as features selecionadas em um csv
def save_selected_features_to_csv(features_selected_indices, data_formatada, original_columns):
    # Usar os índices para obter os nomes das colunas selecionadas
    features_selected_names = original_columns[features_selected_indices]
    pd.DataFrame(features_selected_names, columns=['atributos_selecionados']).to_csv(f'features_selected_{data_formatada}.csv', index=False)

# Salva as métricas do modelo
def save_metrics(metrics_df):
    # Salvar métricas em CSV
    with open('results/metricas_execucoes.csv', 'a') as f:
        metrics_df.to_csv(f, index=False, header=f.tell()==0)

# Move os arquivos para a pasta da execucao
def organize_docs(data_formatada, dataset_name):
    folder_name = (f'Execucao_{data_formatada} - {dataset_name}')
    os.makedirs(folder_name)

    # Teste
    shutil.move(f'confusion_matrix_SVM_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_SVM_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'confusion_matrix_RF_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_RF_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'confusion_matrix_NN_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_NN_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'models_metrics_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'models_metrics_{data_formatada}_{dataset_name}.png'))

    # PROD
    shutil.move(f'confusion_matrix_SVM_prod_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_SVM_prod_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'confusion_matrix_RF_prod_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_RF_prod_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'confusion_matrix_NN_prod_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'confusion_matrix_NN_prod_{data_formatada}_{dataset_name}.png'))
    shutil.move(f'models_metrics_prod_{data_formatada}_{dataset_name}.png', os.path.join(folder_name, f'models_metrics_prod_{data_formatada}_{dataset_name}.png'))

    shutil.move(f'features_selected_{data_formatada}.csv', os.path.join(folder_name, f'features_selected_{data_formatada}.csv'))


    destination_folder = 'results'
    shutil.move(folder_name, destination_folder)

    print(f"Arquivos organizados com sucesso.")

# Função para pré-processar colunas de datas, transformando em ano, mês e dia
def preprocess_dates(df, date_columns):
    for col in date_columns:        
        # Inputa valores ausentes com a data padrão '01/01/1900'
        df[col].fillna("01/01/1900", inplace=True)

        # df[col + '_ano'] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce', dayfirst=True).dt.year # Não se utiliza ANO para não limitar o aprendizado do modelo exclusivamente aos anos do dataset
        df[col + '_mes'] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce', dayfirst=True).dt.month
        df[col + '_dia'] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce', dayfirst=True).dt.day

    df = df.drop(columns=date_columns)
    return df


'''
Funções de avaliação dos modelos
'''

# Função para retornar métricas
def process_metrics(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    return accuracy, precision, recall, f1

# Função para plotar matriz de confusão
def plot_confusion_matrix(model, model_name, X_test, y_test, filename):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=[False, True], yticklabels=[False, True])
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusão {model_name}')
    plt.savefig(filename)
    plt.close()
    return cm

# Função para plotar métricas de precisão, recall, acurácia e f1-score
def plot_metrics(metrics_dict, file_path):
    # Configurações do gráfico
    fig, ax = plt.subplots(figsize=(16, 6))  # Ajusta o tamanho
    bar_width = 0.2
    index = range(len(metrics_dict['Acurácia']))
    models_names = ['SVM', 'RF', 'NN']

    # Plotando as barras
    bars1 = ax.bar(index, metrics_dict['Acurácia'], bar_width, label='Acurácia')
    bars2 = ax.bar([i + bar_width for i in index], metrics_dict['Precisão'], bar_width, label='Precisão')
    bars3 = ax.bar([i + 2 * bar_width for i in index], metrics_dict['Revocação'], bar_width, label='Revocação')
    bars4 = ax.bar([i + 3 * bar_width for i in index], metrics_dict['F1-Score'], bar_width, label='F1-Score')

    # Adicionando os percentuais em cima de cada barra
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')

    # Configurações dos eixos e título
    ax.set_xlabel('Métricas')
    ax.set_ylabel('Valores')
    ax.set_xticks([i + 1.5 * bar_width for i in index])
    ax.set_xticklabels([f'Modelo {models_names[i]}' for i in index])
    ax.set_yticks([i * 0.1 for i in range(int(max(max(metrics_dict.values())) / 0.1) + 1)])

    # Movendo a legenda para baixo do gráfico
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

    # Salvando o gráfico
    plt.tight_layout()
    plt.savefig(file_path)
