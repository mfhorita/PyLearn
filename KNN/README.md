{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <font color='blue'>K-Nearest Neighbors Classifier</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 0. Dependências"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Introdução "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "O KNN (K Nearest Neighbor) é um dos algoritmos mais utilizados em Machine Learning e também um dos mais simplistas. Seu método de aprendizagem é baseado em instâncias e assume que os dados tendem a estar concentrados em uma mesma região no espaço de entrada."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris = load_iris()\n",
    "iris.target"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Dados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                5.1               3.5                1.4               0.2   \n",
       "1                4.9               3.0                1.4               0.2   \n",
       "2                4.7               3.2                1.3               0.2   \n",
       "3                4.6               3.1                1.5               0.2   \n",
       "4                5.0               3.6                1.4               0.2   \n",
       "\n",
       "    class  \n",
       "0  setosa  \n",
       "1  setosa  \n",
       "2  setosa  \n",
       "3  setosa  \n",
       "4  setosa  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame(data=iris.data, columns=iris.feature_names)\n",
    "df['class'] = iris.target\n",
    "df['class'] = df['class'].map({0:iris.target_names[0], 1:iris.target_names[1], 2:iris.target_names[2]})\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.057333</td>\n",
       "      <td>3.758000</td>\n",
       "      <td>1.199333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.435866</td>\n",
       "      <td>1.765298</td>\n",
       "      <td>0.762238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       sepal length (cm)  sepal width (cm)  petal length (cm)  \\\n",
       "count         150.000000        150.000000         150.000000   \n",
       "mean            5.843333          3.057333           3.758000   \n",
       "std             0.828066          0.435866           1.765298   \n",
       "min             4.300000          2.000000           1.000000   \n",
       "25%             5.100000          2.800000           1.600000   \n",
       "50%             5.800000          3.000000           4.350000   \n",
       "75%             6.400000          3.300000           5.100000   \n",
       "max             7.900000          4.400000           6.900000   \n",
       "\n",
       "       petal width (cm)  \n",
       "count        150.000000  \n",
       "mean           1.199333  \n",
       "std            0.762238  \n",
       "min            0.100000  \n",
       "25%            0.300000  \n",
       "50%            1.300000  \n",
       "75%            1.800000  \n",
       "max            2.500000  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(iris.data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(150, 4) (150, 1)\n"
     ]
    }
   ],
   "source": [
    "x = iris.data\n",
    "y = iris.target.reshape(-1, 1)\n",
    "\n",
    "print(x.shape, y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(105, 4) (105, 1)\n",
      "(45, 4) (45, 1)\n"
     ]
    }
   ],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)\n",
    "\n",
    "print(x_train.shape, y_train.shape)\n",
    "print(x_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Implementação"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Métricas de Distância"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Distância de Manhattan\n",
    "def l1_distance(a, b):\n",
    "    return np.sum(np.abs(a - b), axis=1)\n",
    "\n",
    "#Distância Eucladiana\n",
    "def l2_distance(a, b):\n",
    "    return np.sqrt(np.sum(pow(a - b, 2), axis=1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Classificador"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "class kNearestNeighbor(object):\n",
    "    def __init__(self, n_neighbors=1, dist_func=l1_distance):\n",
    "        self.n_neighbors = n_neighbors\n",
    "        self.dist_func = dist_func\n",
    "\n",
    "    def fit(self, x, y):\n",
    "        self.x_train = x\n",
    "        self.y_train = y\n",
    "\n",
    "    def predict(self, x):\n",
    "        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)\n",
    "\n",
    "        for i, x_test in enumerate(x):\n",
    "            distances = self.dist_func(self.x_train, x_test)\n",
    "            nn_index = np.argsort(distances)\n",
    "            nn_pred = self.y_train[nn_index[:self.n_neighbors]].ravel()\n",
    "            y_pred[i] = np.argmax(np.bincount(nn_pred))\n",
    "\n",
    "        return y_pred"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Teste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acurácia: 93.33%\n"
     ]
    }
   ],
   "source": [
    "knn = kNearestNeighbor(n_neighbors=3)\n",
    "knn.fit(x_train, y_train)\n",
    "\n",
    "y_pred = knn.predict(x_test)\n",
    "\n",
    "print('Acurácia: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>k</th>\n",
       "      <th>dist. func.</th>\n",
       "      <th>acurácia</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>l1_distance</td>\n",
       "      <td>91.111111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>l1_distance</td>\n",
       "      <td>93.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>l1_distance</td>\n",
       "      <td>93.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>l2_distance</td>\n",
       "      <td>93.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>l2_distance</td>\n",
       "      <td>95.555556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>l2_distance</td>\n",
       "      <td>97.777778</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   k  dist. func.   acurácia\n",
       "0  1  l1_distance  91.111111\n",
       "1  3  l1_distance  93.333333\n",
       "2  5  l1_distance  93.333333\n",
       "3  1  l2_distance  93.333333\n",
       "4  3  l2_distance  95.555556\n",
       "5  5  l2_distance  97.777778"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn = kNearestNeighbor()\n",
    "knn.fit(x_train, y_train)\n",
    "\n",
    "list_res = []\n",
    "for p in [1, 2]:\n",
    "    knn.dist_func = l1_distance if p == 1 else l2_distance   \n",
    "    \n",
    "    for k in range(1, 6, 2):\n",
    "        knn.n_neighbors = k\n",
    "        y_pred = knn.predict(x_test)\n",
    "        acc = accuracy_score(y_test, y_pred)*100\n",
    "        list_res.append([k, 'l1_distance' if p == 1 else 'l2_distance', acc])\n",
    "        \n",
    "df = pd.DataFrame(list_res, columns=['k', 'dist. func.', 'acurácia'])\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "previcoes = np.array([[6.7,3.1,4.4,1.4],[4.6,3.2,1.4,0.2],[4.6,3.2,1.4,0.2],[6.4,3.1,5.5,1.8],[6.3,3.2,5.6,1.9]])\n",
    "type(previcoes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [0],\n",
       "       [0],\n",
       "       [2],\n",
       "       [2]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fazendo previsões para 5 novas plantas com K igual a 3\n",
    "knn.n_neighbors = 3\n",
    "result = knn.predict(previcoes)\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1],\n",
       "       [0],\n",
       "       [0],\n",
       "       [2],\n",
       "       [2]])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fazendo previsões para 5 novas plantas com K igual a 5\n",
    "knn.n_neighbors = 5\n",
    "result = knn.predict(previcoes)\n",
    "result"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Salvando Modelos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Model_KNeighborsClassifier_Iris_os.sav']"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Vamos salvar o modelo usando o joblib\n",
    "import joblib\n",
    "filename = 'KNeighborsClassifier_Iris_oc.sav' \n",
    "joblib.dump(knn, filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "97.77777777777777\n"
     ]
    }
   ],
   "source": [
    "#Algumas horas ou talvez dias dias depois...\n",
    "#Carrega o modelo do disco\n",
    "loaded_model = joblib.load(filename)\n",
    "y_pred = loaded_model.predict(x_test)\n",
    "result = accuracy_score(y_test, y_pred)*100\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# Fim"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
