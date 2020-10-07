#!/usr/bin/env python
# coding: utf-8

# 
# ![ ](fraude.png)
# _______________________________________________________
# 

# ## Introdução
# 
# Estima-se que em 2018, quase [25 bilhões de dólares](https://shiftprocessing.com/credit-card-fraud-statistics/) tenham-se perdido devido a fraude com cartões em todo o mundo. O Brasil é o [2º país](https://valorinveste.globo.com/produtos/servicos-financeiros/noticia/2020/02/12/brasil-e-2o-pais-da-america-latina-com-mais-fraudes-no-cartao-em-compras-online.ghtml) da América Latina com mais fraudes no cartão em compras online. E em recente estudo divulgado pelo [SPC](https://noticias.r7.com/economia/fraudes-com-cartao-disparam-na-pandemia-veja-como-se-prevenir-19062020), nos últimos 12 meses 9 milhões de brasileiros foram vítimas de fraude, e durante a pandemia, os números se intensificaram.
# 
# E neste projeto, demonstro como funciona o algoritmo de Machine Learning para deteccção de anomalias, neste caso, o evento de anomalia é uma fraude em cartão de crédito.

# ## Base de Dados
# 
# O conjunto de dados contêm transações realizadas por cartões de crédito em setembro de 2013 por titulares de cartões europeus. O conjunto de dados apresenta as transações que ocorreram em dois dias, onde temos 492 fraudes em 284.807 transações. Os dados estão altamentos desbalanceados, o que torna o desafio de predição do evento maior.
# 
# Os dados são apenas variáveis de entrada numéricas e por questões de confidencialidade não estão descritas maiores informações sobre cada uma, os únicos recursos que não foram anonimizados foram as de Tempo e de Valor da transação, e claro, a variável de classe que responde se a transação foi fraude ou não.
# 
# Acesse aqui para [Download](https://www.kaggle.com/mlg-ulb/creditcardfraud), além do Download pode-se encontrar maiores informações a respeito dos dados.

# ### Carregando as bibliotecas e os dados

# In[1]:


# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest

# Para remover os warnings
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[2]:


# Carregando a base de dados

fraud = pd.read_csv("creditcard.csv", sep = ",")
df = fraud.copy()

fraud.shape[0]


# In[3]:


# Visualizando a tabela dos dados
fraud.head()


# ______________________________________________________

# ### Conhecendo os dados

# In[4]:


# Estatisticas da base

fraud.describe()


# OK, muita informação foi jogada na tela, como houve uma transformação PCA sobre os dados, não faz muito sentido olhar para eles assim.

# In[5]:


# Existem missing values?

fraud.isnull().values.any()


# Ótimo! Não temos nenhum valor faltante no dataset, isso nos diz muito sobre a qualidade dos dados e sobre quanto trabalhoso será a análise.

# In[6]:


# Distribuiçao entre as classes?

print("Percentual de fraudes: {}".format(round(fraud[fraud["Class"] == 1].shape[0]/fraud.shape[0]*100, 3)))
print("Percentual de não fraudes: {}".format(round(fraud[fraud["Class"] == 0].shape[0]/fraud.shape[0]*100, 3))) 


# In[7]:


# Estatisticas para fraudes (1) e nao fraudes (0)

fraud_df = pd.DataFrame(fraud[fraud["Class"] == 1]["Amount"].describe())
fraud_df.columns = ['Valores de Fraude']
fraud_df["Valores sem Fraude"] = pd.Series(fraud[fraud["Class"] == 0]["Amount"].describe())
fraud_df


# ______________________________________________________

# ## Conhecendo a Isolation Forest
# 
# ![ ](forest.png)
# 
# **Isolation Forests** é um algoritmo de detecção de anomalias que faz uso de um conjunto de dados de teste não rotulados com a suposição de que a maioria dos dados são normais. Em outras palavras, IF identificam anomalias/outliers em dados desbalanceadas.
# 
# Consistem na construção de um **ensemble de árvores de isolamento (ITs)** para um determinado conjunto de dados e as observações são consideradas anomalias se **possuírem comprimentos médios mais curtos nas ITs**. Metódo parecido com a da nossa querida Random Forest.
# 
# 
# **Cálculo do Isolation Forest:**
# 
# - O algoritmo da Isolation Forest gera recursivamente subconjuntos no conjunto de dados selecionando aleatoriamente uma variável e, em seguida, selecionando aleatoriamente um valor de divisão para aquela variável;
# 
# - As anomalias precisam de menos subconjuntos aleatórios para serem isoladas em comparação com pontos "normais" no conjunto de dados;
# 
# - Portanto, as anomalias serão os pontos que têm um comprimento de caminho menor na árvore, sendo o comprimento do caminho o número de ramos percorridos a partir do nó raiz.
# 
# 
# ![ ](isolation-forest.jpg)
# 
# 
# Outros métodos de detecção de anomalias, como SVM, RF, KNN e Logit precisam de conjuntos de dados rotulados para formar um classificador, o que pode ser difícil em alguns casos, como esse de Fraude. As florestas de isolamento usam particionamento aleatório que torna mais fácil descobrir anomalias, além de suas condições para separar serem mais simples que a dos demais.
# 
# O Isolation Forest perfoma melhor e com menor tempo e exigência computacional do que outros métodos baseados em distância e densidade.
# 
# O Paper com maiores informações, descrição dos cálculos e informações sobre os autores, pode ser encontrado [aqui](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).
# 
# Este é o link da descrição da Isolation Forest na biblioteca [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
# 
# Você também pode ver o meu artigo completo em: (BOTAR O LINK AQUI)
# 

# ### Preparando os dados e Treinando o Modelo

# In[8]:


# Separando os dados com as variáveis em x e o alvo em y
x = fraud.drop("Class", axis = 1)
y = fraud["Class"]

# Separando entre treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)


# In[9]:


x


# In[10]:


# Treinando o modelo
iForest = IsolationForest(n_estimators = 200, max_samples = len(x), random_state = 42) #Informando parâmetros do modelo

iForest.fit(x_train) #Comando que treina o Modelo


# In[11]:


# Predições para o treinamento e teste

y_pred_train1 = iForest.predict(x_train)
y_pred_test1 = iForest.predict(x_test)


# Como **resultado** o modelo nos retorna -1 para os valores de outliers and 1 para valores normais.

# In[12]:


y_pred_test1


# In[13]:


# Reajustando a previsão para ficar alinhado com a Class da base

y_pred_train1 = np.where(y_pred_train1 == -1, 1, 0)
y_pred_test1 = np.where(y_pred_test1 == -1, 1, 0)


# In[14]:


print(np.count_nonzero(y_pred_train1 == 1))

print(np.count_nonzero(y_train == 1))


# In[15]:


print(np.count_nonzero(y_pred_test1 == 1))

print(np.count_nonzero(y_test == 1))


# No Dataset temos 492 transações fraudulentas, dividimos em 246 para treino e teste, e o modelo retorna 326 e 314 respectivamente, ou seja, temos a primeira vista um bom indicador de perfomance do modelo. Vamos checar de fato na **Avaliação dos Resultados**.

# ______________________________________________________

# ### Avaliação dos Resultados

# In[16]:


print("Base de treinamento")
print("Percentual de fraudes: {}".format(round(accuracy_score(y_train,y_pred_train1)*100, 2)))
print("------------------------------")
print("Base de teste")
print("Percentual de fraudes: {}".format(round(accuracy_score(y_test,y_pred_test1)*100, 2)))


# O modelo obteve uma ótima perfomance, com incriveis 99,75% de acurácia na base de teste, mesmo sem fazer qualquer alteração na base ou no modelo. 
# 
# Mas nem só de acurácia vive o mundo, vamos ver o impacto financeiro das fraudes aprovadas, do custo de oportunidade por ter reprovada compras que não eram fraude, isso é claro, sem considerar custos operacionais para se manter um setor de fraude.
# 
# Para entender este racional vamos olhar para a **Matriz de Confusão**, que apesar de sua aplicação para classes desbalanceadas não funcionem bem, vamos olhar para além da literatura e estatística, vamos olhar para o **Entendimento de Negócio** e ver como os Falso Positivos impactariam financeiramente.

# <img src="matriz.png" style="width:200px;"/>

# In[17]:


# Treinamento

print(confusion_matrix(y_train, y_pred_train1))
print()
print("--------------------------------------------------------------------------------")
print()
print(classification_report(y_train, y_pred_train1))


# In[18]:


# Teste

print(confusion_matrix(y_test, y_pred_test1))
print()
print("--------------------------------------------------------------------------------")
print()
print(classification_report(y_test, y_pred_test1))


# In[19]:


# Essa biblioteca coloca o número já na moeda local, sem que tenhamos que ficar preenchendo a mão
# de acordo com a localização do seu computador

import locale
locale.setlocale( locale.LC_ALL, '' )


# In[20]:


print("Transação Normal")
print("Valor médio: {}".format(locale.currency(round(fraud[fraud["Class"] == 0]["Amount"].describe()[1], 2))))
print("___________________________")
print("Transação Fraudulenta")
print("Valor médio: {}".format(locale.currency(round(fraud[fraud["Class"] == 1]["Amount"].describe()[1], 2))))


# In[21]:


print("Teriamos bloqueado mediamente: {}".format(locale.currency(
    round(fraud[fraud["Class"] == 1]["Amount"].describe()[1]*99, 2), grouping=True)))
print("------------------------------")
print("Ao custo médio de : {}".format(locale.currency(
    round(fraud[fraud["Class"] == 0]["Amount"].describe()[1]*215, 2), grouping=True)))


# Estamos bloqueando um pequeno valor a um pequeno custo médio, mas vamos seguir com a análise para vermos outros pontos.

# In[22]:


# Vamos olhar o valor real para a base inteira

# aqui usei o mesmo dataset mas como nome de "df", simplesmente para não impactar o dataset original.

df["IF"] = np.where(iForest.predict(df.iloc[:, 0:30]) == -1, 1, 0)


# In[23]:


locale.currency(df[df["Class"] == 1]["Amount"].sum())


# In[24]:


locale.currency(df[df["IF"] == 1]["Amount"].sum())


# In[25]:


print("Teríamos bloqueado: {}".format(locale.currency(df[(df["IF"] == 1) & (df["Class"] == 1)]["Amount"].sum()
                                                      , grouping=True)))
print()
print("Que representa % do total: {}".format(round(df[(df["IF"] == 1) & (df["Class"] == 1)]["Amount"].sum()/
                                              df[df["Class"] == 1]["Amount"].sum()*100, 2)))
print()
print("Autorizando indevidamente: {}".format(locale.currency(df[(df["IF"] == 0) & (fraud["Class"] == 1)]["Amount"].sum()
                                                    , grouping=True)))
print("------------------------------")
print("Ao custo de: {}".format(locale.currency(df[(df["IF"] == 1) & (df["Class"] == 0)]["Amount"].sum(), grouping=True)))
print("")
print("Que representa % do total: {}".format(round(df[(df["IF"] == 0) & (df["Class"] == 0)]["Amount"].sum()/
                                              df[df["Class"] == 0]["Amount"].sum()*100, 2)))


# O bloco de código nos diz o seguinte:
# 
# * Onde a IF disse que era Fraude e era realmente Fraude, teríamos bloqueado 15.353,35.
# 
# * Apenas 25,5 % das Fraudes totais.
# 
# * Perderiamos 44.774,62 em Fraudes.
# 
# * Onde a IF disse que era fraude, mas na verdade não era, teríamos nos causado uma dor de cabeça de 805.646,24 em transações
# 
# * A Porcentagem de 96,79% é a taxa de acerto da If para transações não fraudulentas
# 
# 
# Seguimos para um modelo mais eficiente.

# ______________________________________________________

# ### Um modelo mais eficiente!

# In[30]:


# Separando os dados com as variáveis em x e o alvo em y
x = df.drop(["Class","IF"], axis = 1)
y = df["Class"]

# Separando entre treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)


# In[31]:


# Treinando o modelo

iForest2 = IsolationForest(n_estimators = 1, max_samples = 100000, random_state = 42)

iForest2.fit(x_train)


# In[32]:


# Prediçoes para o treinamento e teste

y_pred_train2 = iForest2.predict(x_train)
y_pred_test2 = iForest2.predict(x_test)

# Returns -1 for outliers and 1 for inliers.


# In[33]:


# Returns -1 for outliers and 1 for inliers.

# Reajustando a previsao para ficar alinhado com a Class da base

y_pred_train2 = np.where(y_pred_train2 == -1, 1, 0)
y_pred_test2 = np.where(y_pred_test2 == -1, 1, 0)


# ______________________________________________________

# ### Avaliação dos novos resultados

# In[34]:


print("Base de treinamento")
print("Percentual de fraudes: {}".format(round(accuracy_score(y_train,y_pred_train2)*100, 2)))
print("------------------------------")
print("Base de teste")
print("Percentual de fraudes: {}".format(round(accuracy_score(y_test,y_pred_test2)*100, 2)))


# In[36]:


# Mas qual o valor real?

df["IF2"] = np.where(iForest2.predict(df.iloc[:, 0:30]) == -1, 1, 0)


# In[37]:


print("Teriamos bloqueado: {}".format(locale.currency(df[(df["IF2"] == 1) & (df["Class"] == 1)]["Amount"].sum()
                                                      , grouping=True)))
print()
print("% do total: {}".format(round(df[(df["IF2"] == 1) & (df["Class"] == 1)]["Amount"].sum()/
                                              df[df["Class"] == 1]["Amount"].sum()*100, 2)))
print()
print("Deixando passar : {}".format(locale.currency(df[(df["IF2"] == 0) & (df["Class"] == 1)]["Amount"].sum()
                                                    , grouping=True)))
print("------------------------------")
print("Ao custo de : {}".format(locale.currency(df[(df["IF2"] == 1) & (df["Class"] == 0)]["Amount"].sum(), grouping=True)))
print("")
print("% do total ainda aceito: {}".format(round(df[(df["IF2"] == 0) & (df["Class"] == 0)]["Amount"].sum()/
                                              df[df["Class"] == 0]["Amount"].sum()*100, 2)))


# **Relembrando o Resultado anterior:**
# 
# Teríamos bloqueado: R$ 15.353,35
# 
# Que representa  do total: 25.53
# 
# Autorizando indevidamente: 44.774,62
# 
# Ao custo de: R$ 805.646,24
# 
# Que representa % do total: 96.79
# 
# Como podemos ver, ainda que não tivesse sido realizado grande alterações nos parâmetros do modelo, o resultado prever um maior bloqueio de fraudes, e também, uma maior eficácia no que diz a respeito da liberação de transações que de fato são corretas.

# ### Conclusão
# 
# Lidar com dados financeiros é um assunto delicado, ainda mais quando são tão pessoais quanto transações do cartão de crédito, que precisam estar anônimos, o que de certo modo pode dificultar uma análise para predição de fraude. Além disso, a fraude é um evento raro, ou seja, os dados sempre serão desbalanceados, mas não é por ser tratar de um evento raro que não tem importância, conforme mostra os dados utilizados nesta análise, em apenas dois dias houve um prejuízo em torno de R$ 60.000,00 (imaginando que a moeda seria o real).
# 
# O modelo de Isolation Forest, que por definição busca encontrar anomalias e outliers, nos ofereceu uma acurácia de mais de 99 porcento, o que nos mostra que não ter um entendimento do negócio e confiar apenas nos números e na progrmação não servem para resolver problemas do dia a dia. O Algoritmo, por mais que não tenha sio realizado nenhuma grande alteração nos parâmetros, ou mesmos nos dados através de *feature engineering* nos resultou em um bloqueio de 16.000,00, o que representa 25,5 do total das fraudes, um número baixo, mas nos poupou R$ 681.367,22 em dores de cabeça por transações bloqueadas que não deveriam. Talvez não tenha sido o melhor modelo a ser escolhido ou talvez a escassez de dados poderiam resultar em uma análise melhor, mas fica como aprendizado a aplicabilidade ao negócio, e claro, a Isolatio Foret, que apresentou uma boa análise, com velocidade e baixo uso computacional.
# 
