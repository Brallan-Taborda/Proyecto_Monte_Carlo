#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from scipy.stats import norm


# In[2]:


datos = pd.read_csv('USDCOP=X.csv',header=0, usecols=["Date","Close"], parse_dates=['Date'], index_col=0) #datos desde el 19 de febrero de 2016 hasta octubre 4 de 2021
print(datos.shape)


# In[3]:


#plt.figure(figsize=(16,8))
#plt.plot(datos)
#plt.show


# In[4]:


cambioporcentual = datos.pct_change()                          #cambio porcentual de los datos
logaritmo = np.log(1 + cambioporcentual)                       #rendimiento logaritmico


# In[5]:


#plt.figure(figsize=(16,8))
#plt.plot(logaritmo)
#plt.show


# In[6]:


media_logaritmo = np.array(logaritmo.mean())                #media del rendimiento logaritmico
varianza_logaritmo = np.array(logaritmo.var())              #Varianza del rendimiento logaritmico
des_estandar_logaritmo = np.array(logaritmo.std())          #desviacion estandar del rendimiento logaritmico


# In[7]:


drift = media_logaritmo - (0.5*varianza_logaritmo)          #calculo del drift (deriva)
#print("drift= ", drift)


# In[74]:


intervalos = 2000                                           #numero de datos de la simulacion 1469 corresponderian a los datos historicos

N = 5                                                       #numero de simulaciones
 
np.random.seed(16)                                         #semilla para reproductividad
browniano = norm.ppf(np.random.rand(intervalos,N))         #da el valor de la variable para la cual la probabilidad acumulada tiene el valor dado


# In[75]:


datodiario = np.exp(drift + des_estandar_logaritmo*browniano)            #precio diario del dolar usando modelo Black-Scholes


# In[76]:


precio_inicial = datos.iloc[0]                                       #inicializo la cadena con el valor original

precio = np.zeros_like(datodiario)                                   #creo un array de igual tamaño que mis datos

precio[0] = precio_inicial                                          #nuevo valor del dolar (para cada dia)  

for i in range(1, intervalos):                                       #iteracion del precio del dolar
    precio[i] = precio[i-1] * datodiario[i]


# In[77]:


plt.figure(figsize=(8,5))
plt.plot(precio)
tendencia = np.array(datos.iloc[:, 0:1])
plt.xlabel('Número de días')
plt.ylabel('Precio del Dolar (Peso Colombiano)')
plt.xlim(0,intervalos)
plt.plot(tendencia, 'k*')

plt.show


# In[78]:


print(precio.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




