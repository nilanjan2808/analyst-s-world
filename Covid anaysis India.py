#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\\Users\\nilanjan sengupta\\Downloads\\covid_19_india.csv",parse_dates=['Date'],dayfirst=True)
df.head()


# In[3]:


#removing insignificant coloumns and renaming important coloumns for better understanding
df=df[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
df.columns=['date','state','cured','deaths','confirmed']
df.head()


# In[4]:


df.tail()


# In[7]:


today=df[df.date=='2021-06-01']
today


# In[8]:


#sorting with respect to number of confirmed cases
max_confirmed_cases=today.sort_values(by="confirmed",ascending=False)
max_confirmed_cases


# In[10]:


top_states_confirmed=max_confirmed_cases[0:5]
#making a bar plot for states with top confirmed cases 
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="state",y="confirmed",data=top_states_confirmed,hue="state")
plt.show()


# In[11]:


max_death_cases=today.sort_values(by="deaths",ascending=False)
max_death_cases


# In[12]:


#making bar plots of states with the maximum number of death cases
top_states_death=max_death_cases[0:5]
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="state",y="deaths",data=top_states_death,hue="state")
plt.show()


# In[13]:


#Similar process for all the cured cases
#sorting data with respect to the number of cured cases
max_cured_cases=today.sort_values(by="cured",ascending=False)
max_cured_cases


# In[14]:


#extracting the top 5 states again
top_states_cured=max_cured_cases[0:5]
#making the bar plot for states with the top cured cases
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="state",y="cured",data=top_states_cured,hue="state")
plt.show()


# In[15]:


#if the dependant variable is categorical in nature then a classification algorithm will be preffered
maha=df[df.state=='Maharashtra']
maha

#using a line plot for visualized confirmed cases in maharashtra
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=maha,color="g")
plt.show()


# In[19]:


# A similar analysis to find the number of deaths in Maharshtra with respect to date
#visualizing death cases in maharashtra
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="date",y="deaths",data=maha,color="r")
plt.show()


# In[20]:


#same for kerala
kerala=df[df.state=="Kerala"]
kerala
#using a line plot for visualized confirmed cases in Kerala
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=kerala,color="g")
plt.show()
#death cases in Kerala
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="deaths",data=kerala,color="r")
plt.show()


# In[22]:


# Same analysis for Jammu and Kashmir
jk=df[df.state=='Jammu and Kashmir']
jk
# confirmed cases in J&K
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=jk,color="g")
plt.show()
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="deaths",data=jk,color="r")
plt.show()


# In[23]:


#checking the state -wise testing details
tests=pd.read_csv(r"C:\\Users\\nilanjan sengupta\\Downloads\\StatewiseTestingDetails.csv")
tests


# In[24]:


test_latest=tests[tests.Date=='2021-05-30']
test_latest


# In[ ]:


#Sorting the dataset with respect to the number of tests conducted by the states
max_tests_State=test_latest.sort_values(by="TotalSamples",ascending=False)
max_tests_State


# In[ ]:


#A bar plot for states with max_cases
sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="State",y="TotalSamples",data=max_tests_State[0:5],hue="State")
plt.show()


# In[25]:


# Predicting the future number of cases in Maharashtra via Linear regression
#Linear Regression
from sklearn.model_selection import train_test_split
#converting the date entity to ordinal. Because Ordinal basically means numbers. Linear Regression
#algorithm cannot be used on top of date columns. It needs numerical entities
maha['date']=maha['date'].map(dt.datetime.toordinal)
maha.head()


# In[27]:


#creating a dependant variable and an independant variable
x=maha['date']
y=maha['confirmed']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3) #means that 30% of the record go into the test set
#while 70% of the records go into the training set


# In[28]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
x_train


# In[29]:


y_train


# In[30]:


#x_train and y_train are present in the form of series object. Theyre index. 
#  But these values are undesired while fitting a linear regresson model. Therefore we have to use
# np.array () to convert it into single dimension
lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))


# In[31]:


lr.predict(np.array([[737949]]))


# In[32]:


maha.tail()


# In[ ]:


#this number is the ordinal form after converting the date ot ordinal values
#this number is 737949 = 737942(1st June + 7 days) which is 8th may
#There will be 40,53,589 cases on 8th May in Maharashtra

