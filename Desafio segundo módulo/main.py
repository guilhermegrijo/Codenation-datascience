#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday


# In[5]:


black_friday.columns = [col.lower() for col in black_friday]


# In[6]:


black_friday.info(verbose=True)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    # O método .shape retorna retorna observações e colunas em formato de tuple
    return black_friday.shape    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    # A função len retorna o número de colunas de um DataFrame de forma mais perfomática
    return len(black_friday[(black_friday['gender'] == "F") & (black_friday['age'] == '26-35')])    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[9]:


def q3():
    # A coluna user_id tem os ID dos usuários e a função nunique() retorna o numero de elementos unicos
    return black_friday['user_id'].nunique()    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[10]:


def q4():
    # A propriedade dtypes retorna os tipos de dados do dataset e a função nunique() retorna o numero de elementos unicos
    return black_friday.dtypes.nunique()    


# # Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():
    # O registro precisa ter ao menos uma coluna null, a função dropna remove o registro que tenha ao menos um valor null
    # O calculo é feito por parte dividido pelo todo
    return 1- len(black_friday.dropna()) / len(black_friday)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    # isnull() com a função sum() retorna a quantidade de valores null de cada coluna e o max() pega o maior valor entre elas
    return black_friday.isnull().sum().max()
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[13]:


def q7():
    # .mode() seleciona o valor mais frequente, a moda
    return float(black_friday['product_category_3'].mode())
    pass
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[20]:


def q8():
    # Normalizar é reescalar valores para entre 0 e 1, a é função = x - x_min / x_max - x_min
    x = black_friday['purchase']
    return ((x - x.min()) / (x.max() - x.min())).mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[29]:


def q9():    
    # Padronizar é reescalar valores para entre -1 e 1, a é função = x - x_media / desvio_padrao_x
    x = black_friday['purchase']
    std = (x - x.mean()) / x.std()
    return int(((std >= -1) & (std <= 1)).sum())
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[26]:


def q10():
    # Filtra se existe valor em product_category_3 quando product_category_2 é null
    return black_friday[black_friday['product_category_2'].isnull() & black_friday['product_category_2'].notna()].empty

q10()


# In[ ]:




