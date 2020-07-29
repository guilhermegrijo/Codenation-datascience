#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[23]:


import pandas as pd
from math import sqrt
import numpy as np
import seaborn as sns
import sklearn as sk
import scipy.stats as sct
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[ ]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[27]:


countries = pd.read_csv("countries.csv")


# In[28]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(20)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[30]:


countries['Region'].value_counts()
cols = ["Pop_density","Coastline_ratio","Net_migration","Infant_mortality","Literacy","Phones_per_1000","Arable","Crops","Other","Climate","Birthrate","Deathrate","Agriculture","Industry","Service"]
countries[cols] = countries[cols].apply(lambda x: x.str.replace(',','.').astype(float))
countries.head(20)


# In[ ]:


countries.info()


# In[ ]:


countries.Net_migration.describe()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[ ]:


def q1():
    df_striped = countries['Region'].str.strip()
    return list(df_striped.sort_values().unique())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    pop_density = countries["Pop_density"]      
    pop_density = pop_density.values.reshape((-1,1))
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(pop_density)
    bins = discretizer.transform(pop_density)
        
    return (sum(bins[:, 0] == 9))


q2()


# In[ ]:


# Adicionar a moda a coluna Climate separado por regiao

#region_climate = countries.groupby(['Region'])['Climate'].agg(pd.Series.mode)

#countries['Climate'] = countries.groupby('Region')['Climate'].apply(lambda x: x.fillna(x.mode()[0]))

#countries.head(20)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[ ]:


def q3():
    region_climate = countries.groupby(['Region'])['Climate'].agg(pd.Series.mode)

    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)

    course_encoded = one_hot_encoder.fit_transform(countries[["Region","Climate"]].fillna({'Climate': 0}))


    return course_encoded.shape[1]
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[ ]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[ ]:


def q4():

    num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
    ])

    test_country_df = pd.DataFrame([test_country], columns = countries.columns)

    cols = countries.select_dtypes(include=['float64','int64']).columns


    columnTransformer = ColumnTransformer(transformers = [("numerical", num_pipeline, cols)], n_jobs=1)

    columnTransformer.fit(countries)

    result = columnTransformer.transform(test_country_df)[0]

    test_country_df = pd.DataFrame([result], columns = cols)

    return round(test_country_df["Arable"][0],3)

q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[44]:


def q5():

    Net_migration = countries['Net_migration']

    q1 = Net_migration.quantile(.25)

    q3 = Net_migration.quantile(.75)
    
    iqr = q3 - q1
    
    outliers = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    print(f"Faixa considerada \"normal\": {outliers}")

    outliers_abaixo = (Net_migration < outliers[0]).sum()

    outliers_cima = (Net_migration > outliers[1]).sum()

    remove = bool((outliers_abaixo + outliers_cima) / len(Net_migration) > 0.5)

    result = (outliers_abaixo, outliers_cima, remove)

    Net_migration.plot.box()

    return tuple(result)



q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[52]:


from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[62]:


def q6():
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)

    word_sum = newsgroups_counts[:,count_vectorizer.vocabulary_['phone']].sum()
        
    return word_sum

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[69]:


def q7():

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_vectorizer.fit(newsgroup.data)

    newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroup.data)

    tfidf = newsgroups_tfidf_vectorized[:,tfidf_vectorizer.vocabulary_['phone']].sum()

    return round(tfidf,3)

q7()


# In[ ]:




