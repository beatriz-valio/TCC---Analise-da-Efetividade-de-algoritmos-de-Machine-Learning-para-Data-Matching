# "Análise da efetividade dos algoritmos de machine learning mais citados na literatura dos últimos dez anos para aplicações de data matching."
## Beatriz Valio Weiss, 2024

Código desenvolvido ao longo do Trabalho de Conclusão de Curso na Universidade Federal de Santa Catarina, curso de Sistemas de Informação pela aluna Beatriz Valio Weiss.

## Objetivo
Obter métricas de desempenho dos 03 algoritmos mais citados na literatura, aplicados na etapa de Classificação  

## Saiba mais
Para mais informações sobre o trabalho realizado, acesse o trabalho completo na (biblioteca da UFSC)[].

## Desenvolvimento e Execução

### Algoritmos implementados
Para avaliação estão incluídos nesse código:
- Random Forest: `RandomForestClassifier`
- Support Vector Machine: `SVC`
- Neural Networks: `MLPClassifier`

### Estruturação do código
1. `main.py`: executa o processo de ponta a ponta e avalia ao final uma simulação de como seria no ambiente de produção, utilizando 20% do conjunto de dados definido para teste. 
1. `utils.py`: contém algumas funções auxiliares para deixar a `main.py` mais organizada.
1. `feature_selection.py`: seleciona automaticamente as features para todos os modelos utilizando `CatBoostClassifier` e `optuna`, selecionado aquelas que tiverem importancia maior que 0.
1. `transformations.py`: transforma cada feature selecionada de acordo com sua classificação na main (numérica, categórica ou textual).
1. `hyperparameter_optimization.py`: seleciona automaticamente os melhores hiperparametros encontrados dentro do numero de tentativas (setado inicialmente 5 tentativas na variável `n_trials`) para os algoritmos utilizando `optuna` e `cross-validation`.
1. `models.py`: avalia final dos algoritmos.

#### Resultados
Métricas para comparação dos algoritmos em relação ao seu potencial de classificação para o dataset em questão. 

O resultado da execução do código inclue:
- Salvamento da melhor versão do modelo com os hiperparâmetros encontrados na pasta `/best_models`
- Salvamento das métricas em `results/metricas_execucoes.csv`
- Salvamento das features selecionadas e gráficos em `results/Execucao_dd_mm_YY_HH_MM_SS`

### Como executar?
- Ter Python 3.11 instalado
- Instalar as bibliotecas necessárias: `pip install -r requirements.txt`
- Garantir que o dataset esteja em `/data`
- Configurar as particularidades do dataset na `main.py`:
  - `dataset_name`: nome do dataset (arquivo csv) inserido em `/data` (ex.: 'empresas.csv')
  - `delimiter`: delimitador (',' ou ';')
  - `target_column`: atributo alvo, provavelmente nomeado de 'matching' (booleano 'True'/'False')
  - `date_columns`: nome de todos os atributos que possuem data (devendo estar no formato 'dd/mm/aaaa')
  - `numeric_columns`: nome de todos os atributos numéricos (atenção para o separador de decimal ser '.')
  - `categoric_columns`: nome de todos os atributos categóricos
  - `text_columns`: nome de todos os atributos textuais
  - `key_columns`: nome de todos os atributos que apenas identificam o registro nas tabelas originarias (os atributos identificados aqui não devem constar em numeric_columns, categoric_columns e text_columns)
