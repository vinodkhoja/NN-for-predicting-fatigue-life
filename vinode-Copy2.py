#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
 
import os


# In[2]:


#os.chdir("../input")
import pandas as pd
data = pd.read_csv("T351_aluminium_data_true.csv")
data[0:5]


# In[3]:


original_data = pd.read_csv("T351_aluminium_data_true.csv")


# In[4]:


data.shape


# In[5]:


data["crack_growth_rate"] = (10**(5) ) * data["crack_growth_rate"]


# In[6]:


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 


# In[7]:


from keras.models import model_from_json


# In[8]:


x=data.drop(['crack_growth_rate'],axis=1)
y=data.drop(['R','delta_K'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.1)


# In[9]:


X_train.shape[1]


# In[10]:


NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='sigmoid'))


# In[11]:


# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[12]:


NN_model.fit(X_train,y_train, epochs=10000, batch_size=32, validation_split = 0.15)


# In[13]:


predictions = NN_model.predict(X_test)


# In[14]:


import sklearn.metrics as metrics
metrics.explained_variance_score(y_test, predictions)


# In[81]:


predictions


# In[15]:


y_test


# In[16]:


actual_y = y_test.values
actual_y[3] -0


# In[17]:


error =0
for i in range(8):
    error += abs((actual_y[i])-(predictions[i]))
    


# In[18]:


error/8


# In[19]:


data.mean()


# In[20]:


((error/8)/4.14)*100


# In[21]:


percentage_error = 0
total_error = 0
total_per_error = 0
for i in range(8):
    per_error = ((abs(actual_y[i]-predictions[i]))/(actual_y[i]) )* 100
    percentage_error += per_error
    


# In[22]:


percentage_error / 8


# In[96]:


X_test


# In[38]:


my_X = data.drop(['crack_growth_rate'],axis=1)
y=data.drop(['R','delta_K'],axis=1)
predicting_on_complete_dataset = NN_model.predict(my_X)


# In[39]:


predicting_on_complete_dataset.shape
my_y = y.values


# In[40]:


percentage_error = 0
total_error = 0
total_per_error = 0
for i in range(77):
    per_error = ((abs(my_y[i]-predicting_on_complete_dataset[i]))/(my_y[i]) )* 100
    percentage_error += per_error


# In[41]:


percentage_error/77


# In[33]:


model_json = NN_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
NN_model.save_weights("model.h5")
print("Saved model to disk")


# In[34]:


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[42]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection = '3d')

X = data['R'].values
Y = data['delta_K'].values
Z = data['crack_growth_rate'].values
ax.scatter(X,Y,Z, c = 'b' , marker = 'o')
ax.set_xlabel('xaxis')
ax.set_ylabel('yaxis')
ax.set_zlabel('z_axis')
plt.show()


# In[43]:


subdf_0 = data.loc[data['R'] == 0]
subdf_1 = data.loc[data['R'] == 0.1]
subdf_3 = data.loc[data['R'] == 0.3]
subdf_5 = data.loc[data['R'] == 0.5]


# In[44]:


subdf_0


# In[46]:


## predict for R = 0
r0_data_predictions = loaded_model.predict(subdf_0.drop("crack_growth_rate",axis = 1))


# In[49]:


fig = plt.figure(figsize=(12,12))


ax0= fig.add_subplot(111)
ax0.scatter(subdf_0.delta_K, r0_data_predictions, c= 'b',marker ='.')
ax0.scatter(subdf_0.delta_K,subdf_0.crack_growth_rate, c= 'r' , marker ='.')
# ax0.scatter(subdf_1.delta_K,subdf_1.crack_growth_rate, c= 'b' , marker ='.')
# ax0.scatter(subdf_3.delta_K,subdf_3.crack_growth_rate, c= 'b' , marker ='o')
# ax0.scatter(subdf_5.delta_K,subdf_5.crack_growth_rate, c= 'r' , marker ='o')


# In[ ]:


# one hell of a result/.///chill


# In[73]:


np.tan(np.pi)


# In[87]:


temp_df = pd.DataFrame(data=None, columns=data.columns)
temp_df = temp_df.drop("crack_growth_rate", axis = 1)
new_row = {'R':0, 'delta_K':0.29}
temp_df = temp_df.append(new_row, ignore_index=True)
temp_df


# In[ ]:


## fatigue life calculation(in cycles)
# check the units and dimensions before calculating
a = 0.5
if(R==0):
    delta_sigma = 80 #in MPa
else if(R == 0.1):
    delta_sigma = 77
else if(R == 0.3):
    delta_sigma = 53
else if(R == 0.5 ):
    delta_sigma = 46
width = 7.2 #in mm
while( a < 5 ):
    temp_df = pd.DataFrame(data=None, columns=data.columns)
    temp_df = temp_df.drop("crack_growth_rate",axis = 1)
    
    # formula from Niel_Anderson for mode-1 laoding
    Y = (width/a) * (1 /(np.pi)) * (np.tan ( (np.pi)/(width/a)))#calculating geometric factor
    
    temp_delta_k = Y * delta_sigma * np.sqrt( np.pi * a) #calculating delta k for this cycle
    
    new_row = {'R':R, 'delta_K':temp_delta_k}
    temp_df = temp_df.append(new_row, ignore_index=True)
    da = loaded_model.predict(temp_df)
    a += da
    

