import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api

'''
Transforma features selecionadas
'''

# Carregar o modelo Word2Vec pré-treinado
word2vec_model = api.load("word2vec-google-news-300")

# Função para converter texto em embeddings usando Word2Vec
def word2vec_transformer(text):
    tokens = str(text).split()  # Dividir o texto em palavras (tokens)
    word_vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]

    # Se não houver palavras reconhecidas no modelo, retornar vetor nulo
    if len(word_vectors) == 0:
        return np.zeros(300)
    
    # Fazer a média dos vetores para representar o texto como um único vetor
    return np.mean(word_vectors, axis=0)

# Função personalizada para processar colunas de texto em embeddings Word2Vec
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        word2vec_embeddings = []
        for col in self.text_columns:
            col_data = X[col] if isinstance(X, pd.DataFrame) else X[:, col]
            embeddings = np.vstack([word2vec_transformer(text) for text in col_data])
            word2vec_embeddings.append(embeddings)
        return np.hstack(word2vec_embeddings)

    def get_feature_names_out(self, input_features=None):
        feature_names = []
        for col in self.text_columns:
            feature_names.extend([f"{col}_word2vec_{i}" for i in range(300)])
        return feature_names

# Função para obter um ColumnTransformer com os transformadores definidos
def get_preprocessor(numeric_columns, categorical_columns, text_columns):
    # Pipeline para processar colunas numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputar valores numéricos ausentes
        ('scaler', StandardScaler())  # Normalizar
    ])

    # Pipeline para processar colunas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputar valores categóricos ausentes
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificação categórica
    ])

    # Transformer para processar colunas de texto usando Word2Vec
    word2vec_transformer = Word2VecTransformer(text_columns=text_columns)

    # Combinar transformadores numéricos, categóricos e textuais
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),  # Pré-processamento de variáveis numéricas
            ('cat', categorical_transformer, categorical_columns),  # Pré-processamento de variáveis categóricas
            ('txt', word2vec_transformer, text_columns)  # Pré-processamento de variáveis textuais com Word2Vec
        ],
        remainder='drop'  # Descartar colunas não especificadas
    )

    return preprocessor

# Função para aplicar a transformação e retornar as variáveis separadamente
def apply_transformation_all_features(X_full, features_index_selected, numeric_columns, categoric_columns, text_columns):
    transformed_data = []

    # Obter os nomes das colunas selecionadas a partir dos índices
    selected_columns = X_full.columns[features_index_selected].tolist()

    # Verificar e transformar colunas numéricas
    valid_numeric_columns = [col for col in numeric_columns if col in selected_columns]
    print("Colunas numéricas válidas:", valid_numeric_columns)
    if valid_numeric_columns:
        preprocessor_numeric = get_preprocessor(numeric_columns=valid_numeric_columns, categorical_columns=[], text_columns=[])
        X_transf_numeric = preprocessor_numeric.fit_transform(X_full[valid_numeric_columns])
        transformed_data.append(X_transf_numeric)
        print('Numéricos transformados com sucesso.')
    else:
        print('Nenhuma coluna para transformação numérica encontrada.')

    # Transformar categóricos se existirem no DataFrame
    valid_categoric_columns = [col for col in categoric_columns if col in selected_columns]
    print("Colunas categóricas válidas:", valid_categoric_columns)
    if valid_categoric_columns:
        preprocessor_categoric = get_preprocessor(numeric_columns=[], categorical_columns=valid_categoric_columns, text_columns=[])
        X_transf_categoric = preprocessor_categoric.fit_transform(X_full[valid_categoric_columns])
        transformed_data.append(X_transf_categoric)
        print('Categóricos transformados com sucesso.')
    else:
        print('Nenhuma coluna categórica encontrada para transformação.')

    # Transformar textuais se existirem no DataFrame
    valid_text_columns = [col for col in text_columns if col in selected_columns]
    print("Colunas textuais válidas:", valid_text_columns)
    if valid_text_columns:
        preprocessor_text = get_preprocessor(numeric_columns=[], categorical_columns=[], text_columns=valid_text_columns)
        X_transf_text = preprocessor_text.fit_transform(X_full[valid_text_columns])
        transformed_data.append(X_transf_text)
        print('Textuais transformados com sucesso.')
    else:
        print('Nenhuma coluna textual encontrada para transformação.')

    # Combinar as transformações
    if transformed_data:
        combined_transformed_data = np.hstack(transformed_data)
    else:
        print("Erro: Nenhuma transformação foi feita. Retornando um array vazio.")
        combined_transformed_data = np.array([])  # Se nenhuma transformação foi feita

    # Retornar o DataFrame combinado e as transformações separadas
    return combined_transformed_data
