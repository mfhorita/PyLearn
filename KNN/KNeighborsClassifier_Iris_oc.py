#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>K-Nearest Neighbors Classifier</font>

# # 0. Dependências

# In[1]:


import joblib
import numpy as np
import pandas as pd
# import sweetviz as sv

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # 1. Introdução 

# O KNN (K Nearest Neighbor) é um dos algoritmos mais utilizados em Machine Learning e também um dos mais simplistas.
# Seu método de aprendizagem é baseado em instâncias e assume que os dados tendem a estar concentrados em uma mesma
# região no espaço de entrada.

# In[2]:

iris = load_iris()
print(iris.target)


# # 2. Dados

# In[3]:


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
df.head(5)

# In[4]:

# Análise Exploratória (sweetviz)
# analise = sv.analyze(df)
# analise.show_html('result.html')

# In[4]:

df.describe()


# In[5]:


type(iris.data)


# In[6]:


x = iris.data
y = iris.target.reshape(-1, 1)

print(x.shape, y.shape)


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# # 3. Implementação

# ### Métricas de Distância

# In[8]:


# Distância de Manhattan
def l1_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


# Distância Eucladiana
def l2_distance(a, b):
    return np.sqrt(np.sum(pow(a - b, 2), axis=1))


# ### Classificador

# In[9]:


class KNearestNeighbor(object):
    def __init__(self, n_neighbors=1, dist_func=l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

        for i, x_test in enumerate(x):
            distances = self.dist_func(self.x_train, x_test)
            nn_index = np.argsort(distances)
            nn_pred = self.y_train[nn_index[:self.n_neighbors]].ravel()
            y_pred[i] = np.argmax(np.bincount(nn_pred))

        return y_pred


# ## 4. Teste

# In[10]:


knn = KNearestNeighbor(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print('Acurácia: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))


# In[11]:


knn = KNearestNeighbor()
knn.fit(x_train, y_train)

list_res = []
for p in [1, 2]:
    knn.dist_func = l1_distance if p == 1 else l2_distance   
    
    for k in range(1, 6, 2):
        knn.n_neighbors = k
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_test, y_pred)*100
        list_res.append([k, 'l1_distance' if p == 1 else 'l2_distance', acc])
        
df = pd.DataFrame(list_res, columns=['k', 'dist. func.', 'acurácia'])
print(df)


# In[12]:


previcoes = np.array([[6.7, 3.1, 4.4, 1.4], [4.6, 3.2, 1.4, 0.2],
                      [4.6, 3.2, 1.4, 0.2], [6.4, 3.1, 5.5, 1.8], [6.3, 3.2, 5.6, 1.9]])
type(previcoes)


# In[13]:


# Fazendo previsões para 5 novas plantas com K igual a 3
knn.n_neighbors = 3
result = knn.predict(previcoes)
print(result)


# In[14]:


# Fazendo previsões para 5 novas plantas com K igual a 5
knn.n_neighbors = 5
result = knn.predict(previcoes)
print(result)


# # 5. Salvando Modelos

# In[15]:


# Vamos salvar o modelo usando o joblib
filename = 'KNeighborsClassifier_Iris_oc.sav'
joblib.dump(knn, filename)


# In[16]:


# Algumas horas ou talvez dias dias depois...
# Carrega o modelo do disco
loaded_model = joblib.load(filename)
y_pred = loaded_model.predict(x_test)
result = accuracy_score(y_test, y_pred)*100
print(result)


# # Fim
