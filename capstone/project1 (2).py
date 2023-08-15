#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df = pd.read_csv('C:\\Users\\User\\Downloads\\Trips_by_Distance.csv')
df.head(5)


# In[2]:


df.duplicated().sum()


# In[3]:


df.isnull().sum()


# In[4]:


df.isnull().sum().sum()


# In[5]:


df["County_FIPS"].mean()


# In[6]:


round(df["County_FIPS"].mean(),0)


# In[7]:


mean_County_FIPS = round(df["County_FIPS"].mean(),0)
mean_County_FIPS


# In[8]:


df["County_FIPS"] = df["County_FIPS"].fillna(mean_County_FIPS)
df["County_FIPS"]


# In[9]:


df["State_FIPS"].mean()


# In[10]:


round(df["State_FIPS"].mean(),0)


# In[11]:


mean_State_FIPS = round(df["State_FIPS"].mean(),0)
mean_State_FIPS


# In[12]:


df["State_FIPS"] = df["State_FIPS"].fillna(mean_County_FIPS)
df["State_FIPS"]


# In[13]:


df.isnull().sum()


# In[14]:


df["Population_Staying_at_Home"].mean()


# In[15]:


#df.dropna()


# In[16]:


df.isnull().sum()


# In[17]:


round(df["Population_Staying_at_Home"].mean(),0)


# In[18]:


mean_Population_Staying_at_Home = round(df["Population_Staying_at_Home"].mean(),0)
mean_Population_Staying_at_Home


# In[19]:


df["Population_Staying_at_Home"] = df["Population_Staying_at_Home"].fillna(mean_Population_Staying_at_Home )
df["Population_Staying_at_Home"]


# In[20]:


df.isnull().sum()


# In[21]:


df["Population_Not_Staying_at_Home"].mean()


# In[22]:


round(df["Population_Not_Staying_at_Home"].mean(),0)


# In[23]:


mean_Population_Not_Staying_at_Home = round(df["Population_Not_Staying_at_Home"].mean(),0)
mean_Population_Not_Staying_at_Home


# In[24]:


df["Population_Not_Staying_at_Home"] = df["Population_Not_Staying_at_Home"].fillna(mean_Population_Not_Staying_at_Home )
df["Population_Not_Staying_at_Home"]


# In[25]:


df.isnull().sum()


# In[26]:


df["Number_of_Trips"].mean()


# In[27]:


round(df["Number_of_Trips"].mean(),0)


# In[28]:


mean_Number_of_Trips = round(df["Number_of_Trips"].mean(),0)
mean_Number_of_Trips


# In[29]:


df["Number_of_Trips"] = df["Number_of_Trips"].fillna(mean_Number_of_Trips )
df["Number_of_Trips"]


# In[30]:


df.isnull().sum()


# In[31]:


df["Number_of_ Trips<1"].mean()


# In[32]:


round(df["Number_of_ Trips<1"].mean(),0)


# In[33]:


mean_Number_of_Trips= round(df["Number_of_ Trips<1"].mean(),0)
mean_Number_of_Trips


# In[34]:


df["Number_of_ Trips<1"] = df["Number_of_ Trips<1"].fillna(mean_Number_of_Trips<1)
df["Number_of_ Trips<1"]


# In[35]:


df["Number_of_Trips_1-3"].mean()


# In[36]:


round(df["Number_of_Trips_1-3"].mean(),0)


# In[37]:


mean_Number_of_Trips_1= round(df["Number_of_Trips_1-3"].mean(),0)
mean_Number_of_Trips_1


# In[38]:


df["Number_of_Trips_1-3"] = df["Number_of_Trips_1-3"].fillna(mean_Number_of_Trips_1)
df["Number_of_Trips_1-3"]


# In[39]:


df["Number_of_Trips_3-5"].mean()


# In[40]:


round(df["Number_of_Trips_3-5"].mean(),0)


# In[41]:


mean_Number_of_Trips_3= round(df["Number_of_Trips_3-5"].mean(),0)
mean_Number_of_Trips_3


# In[42]:


df["Number_of_Trips_3-5"] = df["Number_of_Trips_3-5"].fillna(mean_Number_of_Trips_3)
df["Number_of_Trips_3-5"]


# In[43]:


df["Number_of_Trip_5-10"].mean()


# In[44]:


round(df["Number_of_Trip_5-10"].mean(),0)


# In[45]:


mean_Number_of_Trip_5_10 = round(df["Number_of_Trip_5-10"].mean(),0)
mean_Number_of_Trip_5_10


# In[46]:


df["Number_of_Trip_5-10"] = df["Number_of_Trip_5-10"].fillna(mean_Number_of_Trip_5_10)
df["Number_of_Trip_5-10"]


# In[47]:


df.isnull().sum()


# In[48]:


df["Number_of_Trips(10-25)"].mean()


# In[49]:


round(df["Number_of_Trips(10-25)"].mean(),0)


# In[50]:


mean_Number_of_Trips=round(df["Number_of_Trips(10-25)"].mean(),0)
mean_Number_of_Trips


# In[51]:


df["Number_of_Trips(10-25)"] = df["Number_of_Trips(10-25)"].fillna(mean_Number_of_Trips)
df["Number_of_Trips(10-25)"]


# In[52]:


df.isnull().sum()


# In[53]:


df["Number_of_Trips_25-50"].mean()


# In[54]:


round(df["Number_of_Trips_25-50"].mean(),0)


# In[55]:


mean_Number_of_Trip_25 = round(df["Number_of_Trips_25-50"].mean(),0)
mean_Number_of_Trip_25


# In[56]:


df["Number_of_Trips_25-50"] = df["Number_of_Trips_25-50"].fillna(mean_Number_of_Trip_25)
df["Number_of_Trips_25-50"]


# In[57]:


df.isnull().sum()


# In[58]:


df["Number_of_Trips_50-100"].mean()


# In[59]:


round(df["Number_of_Trips_50-100"].mean(),0)


# In[60]:


mean_Number_of_Trip_50 = round(df["Number_of_Trips_50-100"].mean(),0)
mean_Number_of_Trip_50


# In[61]:


df["Number_of_Trips_50-100"] = df["Number_of_Trips_50-100"].fillna(mean_Number_of_Trip_50)
df["Number_of_Trips_50-100"]


# In[62]:


df.isnull().sum()


# In[63]:


df["Number_of_Trips_100-250"].mean()


# In[64]:


round(df["Number_of_Trips_100-250"].mean(),0)


# In[65]:


mean_Number_of_Trip_100 = round(df["Number_of_Trips_100-250"].mean(),0)
mean_Number_of_Trip_100


# In[66]:


df["Number_of_Trips_100-250"] = df["Number_of_Trips_100-250"].fillna(mean_Number_of_Trip_100)
df["Number_of_Trips_100-250"]


# In[67]:


df["Number_of_Trips_250-500"].mean()


# In[68]:


round(df["Number_of_Trips_250-500"].mean(),0)


# In[69]:


mean_Number_of_Trips_250 = round(df["Number_of_Trips_250-500"].mean(),0)
mean_Number_of_Trips_250


# In[70]:


df["Number_of_Trips_250-500"] = df["Number_of_Trips_250-500"].fillna(mean_Number_of_Trips_250)
df["Number_of_Trips_250-500"]


# In[71]:


df["Number_of_Trips >=500"].mean()


# In[72]:


round(df["Number_of_Trips >=500"].mean(),0)


# In[73]:


mean_Number_of_Trips_500 = round(df["Number_of_Trips >=500"].mean(),0)
mean_Number_of_Trips_500


# In[74]:


df["Number_of_Trips >=500"] = df["Number_of_Trips >=500"].fillna(mean_Number_of_Trips_500)
df["Number_of_Trips >=500"]


# In[75]:


df = df.drop(['County_Name'], axis=1)
display(df.head(5))


# In[76]:


df.isnull().sum()


# In[77]:


## we use copy() here, just to slicing data.
df = df.dropna().copy()
df


# In[78]:


df.info()


# In[79]:


df["State_FIPS"].dtypes


# In[80]:


df["Date"] =  pd.to_datetime(df["Date"],format = "%Y-%m-%d")
df["Date"]


# In[81]:


df["Date"] =  pd.to_datetime(df["Date"],format = "%Y-%m-%d")


# In[82]:


df['Date'].dt.strftime('%Y-%m-%d')


# In[83]:


df.dtypes


# In[84]:


df["Date"] = df["Date"].replace({"low":0,"medium":1,"high":2})


# In[85]:


df["Date"].dtypes


# In[86]:


df.describe()


# In[87]:


df.describe().T


# In[88]:


a=df.describe().T
a
a.to_csv("a.csv")


# In[89]:


df.shape


# In[90]:


df.plot.hist();


# In[91]:


df.plot.scatter(x = 'State_FIPS', y = 'County_FIPS')


# In[92]:


df.plot.scatter(x = 'Number_of_Trips', y = 'Number_of_ Trips<1')


# In[93]:


df.plot.scatter(x = 'Number_of_Trips_1-3', y = 'Number_of_Trips_3-5')


# In[94]:


df.plot.scatter(x = 'Number_of_Trip_5-10', y = 'Number_of_Trips(10-25)')


# In[95]:


df.plot.scatter(x = 'Number_of_Trips_25-50', y = 'Number_of_Trips_100-250')


# In[96]:


df.State_FIPS.plot(kind='hist')


# In[97]:


plt.hist(df["Month"])
plt.title('people_average')
plt.xlabel('how many people stay at home')
plt.ylabel('weekly/monthly')
plt.tight_layout()
plt.show()


# In[98]:


corr_matrix = df.corr()


# In[99]:


round(df.var(),0)


# In[100]:


import seaborn as sns
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm")


# In[101]:


# here, we change Level type because it is object type so , we change it in category type.
df["Level"] = df["Level"].astype("category")


# In[102]:


df["Level"] = df["Level"].replace({'County':1,'National':2,'State': 3})


# In[103]:


df.dtypes


# In[104]:


df1 = df.drop("Date", axis = 1)
df1.head(5)


# In[105]:


df1 = df1.drop("Row_ID", axis = 1)
df1.head(5)


# In[106]:


df1 = df1.drop("State_Postal_Code", axis = 1)
df1.head(5)


# In[107]:


df1 = df1.drop("Number_of_ Trips<1", axis = 1)
df1.head(5)


# In[108]:


df1.isnull().sum()


# In[109]:


from sklearn.preprocessing import MinMaxScaler


# In[110]:


scaler = MinMaxScaler()


# In[111]:


scaler.fit_transform(df1)


# In[112]:


scaler_input = pd.DataFrame(scaler.fit_transform(df1),columns = df1.columns)


# In[113]:


pd.DataFrame(scaler.fit_transform(df1))


# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(scaler_input, df1['Level'], test_size = 0.1, random_state = 6)


# In[116]:


from sklearn.neighbors import KNeighborsClassifier


# In[117]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn


# In[143]:


knn.fit(X_train, y_train)
pred_k = knn.predict(X_test)
pred_k


# In[118]:


clf = DecisionTreeClassifier()


# In[119]:


clf = clf.fit(X_train, y_train)


# In[120]:


#this code runs to get parameters in our dataset.
clf.get_params()


# In[121]:


# by run this code, we get to know how generally our model is predicting on the information is given.
X_test


# In[122]:


#performing entropy 
clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 10)
clf_entropy.fit(X_train, y_train)


# In[123]:


y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)


# In[124]:


# after use this code, we will see the highest probabilty.
predictions = clf.predict(X_test)
predictions


# In[125]:


clf.predict_proba(X_test)


# In[126]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[127]:


print("Accuracy = ", accuracy_score(y_test, y_pred_en)*100)


# In[132]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, predictions, labels=[0,1])
print("cm: ")
print(cm)


# In[138]:


import seaborn as sns 
import matplotlib.pyplot as plt
sns.heatmap(cm,cmap="Greens", annot=True, cbar_kws={"orientation":"vertical","label":"color bar"},
           xticklabels=[0,1],yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[134]:


from sklearn.metrics import precision_score
precision_score(y_test, predictions)


# In[ ]:




