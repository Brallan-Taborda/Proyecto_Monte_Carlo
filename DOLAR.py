#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import norm


# In[33]:


datos = pd.read_csv('USDCOP=X.csv',header=0, usecols=["Date","Close"], parse_dates=['Date'], index_col=0) #datos desde el 2016 hasta octubre 4 2021
#datos = pd.read_csv('FB.csv',header=0, usecols=["Date","Close"], parse_dates=['Date'], index_col=0)


# In[34]:


#print(datos.info())
#print(datos.head())
#print(datos.tail())
#print(datos.describe())
print(datos.shape)


# In[35]:


plt.figure(figsize=(16,8))
plt.plot(datos)
plt.show


# In[7]:


cambioporcentual = datos.pct_change()


# In[8]:


logaritmo = np.log(1 + cambioporcentual)
#print(logaritmo.tail(10))


# In[9]:


plt.figure(figsize=(16,8))
plt.plot(logaritmo)
plt.show


# In[10]:


media_logaritmo = np.array(logaritmo.mean())
varianza_logaritmo = np.array(logaritmo.var())
des_estandar_logaritmo = np.array(logaritmo.std())


# In[11]:


drift = media_logaritmo - (0.5*varianza_logaritmo)
print("drift= ", drift)


# In[29]:


intervalos = 2000

N = 10

np.random.seed(10)
browniano = norm.ppf(np.random.rand(intervalos,N))


# In[30]:


datodiario = np.exp(drift + des_estandar_logaritmo*browniano)


# In[31]:


precio_inicial = datos.iloc[0]

precio = np.zeros_like(datodiario)

precio[0] = precio_inicial

for i in range(1, intervalos):
    precio[i] = precio[i-1] * datodiario[i]


# In[32]:


plt.figure(figsize=(16,8))
plt.plot(precio)
tendencia = np.array(datos.iloc[:, 0:1])
plt.plot(tendencia, 'k*')

plt.show


# In[16]:


print(precio.shape)


# In[ ]:





# In[ ]:





# In[ ]:




