
# coding: utf-8

# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns


# In[ ]:


#set working directory
os.chdir("F:\Others\Project Absenteesim")


# In[ ]:


#Load data
data_absent = pd.read_csv("Absenteeism_at_work_project.csv")


# In[ ]:


data_absent.dtypes


# In[ ]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(data_absent.isnull().sum())


# In[ ]:


#Reset Index
missing_val = missing_val.reset_index()


# In[ ]:


#Rename variables
missing_val = missing_val.rename(columns={'index':'Variables', 0 : 'Missing_Percentage'})


# In[ ]:


#calculate percentage
missing_val["Missing_Percentage"] = (missing_val['Missing_Percentage']/len(data_absent))*100


# In[ ]:


#desecnding order 
missing_val = missing_val.sort_values('Missing_Percentage', ascending = False).reset_index(drop = True)


# In[ ]:


data_absent.head(10)


# In[ ]:


data_absent["Disciplinary failure"].loc[70]


# In[ ]:


data_absent["Disciplinary failure"].loc[70] = np.nan


# In[ ]:


#KNN Imputation
#Assigining level to categories
for i in range(0, data_absent.shape[1]):
    #print(i)
    if(data_absent.iloc[:,i].dtypes == 'object'):
        data_absent.iloc[:,i] = pd.Categorical(data_absent.iloc[:,i])
        #Print(marketing_train[[i]])
        data_absent.iloc[:,i] = data_absent.iloc[:,i].cat.codes


# In[ ]:


data_absent.head(10)


# In[ ]:


#KNN imputation
data_absent = pd.DataFrame(KNN(k = 7).complete(data_absent), columns = data_absent.columns)


# In[ ]:


data_absent.head(10)


# In[ ]:


data_absent["Disciplinary failure"].loc[70]


# In[ ]:


#plot boxplot to visualize outliers
get_ipython().run_line_magic('matplotlib', 'inline')

plt.boxplot(data_absent['Transportation expense'])


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(data_absent['Transportation expense'], [75 ,25])


# In[ ]:


#Calculate Iqr interquartile range
iqr = q75 - q25


# In[ ]:


#calculate inner and outer fence
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)


# In[ ]:


#Replace values which are above and below the fence with NA
data_absent.loc[data_absent['Transportation expense'] < min,:'Transportation expense'] = np.nan
data_absent.loc[data_absent['Transportation expense'] > max,:'Transportation expense'] = np.nan


# In[ ]:


#Calculate missing value
missing_val_absent = pd.DataFrame(data_absent.isnull().sum())


# In[ ]:


#Impute with KNN
data_absent = pd.DataFrame(KNN(k = 7).complete(data_absent), columns = data_absent.columns)


# In[ ]:


data_absent.isnull().sum()


# In[ ]:


data_absent.head(5)


# In[ ]:


#Save numeric variables
cnames = ["ID","Transportation Expense", "Distance from residence to work","Service time","Age","Weight","Height","Body mass index","Absenteeism time in hours"]


# In[ ]:


##Correlation plot
#Corelation plot
df_corr = data_absent.loc[:,cnames]


# In[ ]:


#Set the width and height of plot
f , ax = plt.subplots(figsize =(7,5))

#Set correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask = np.zeros_like(corr ,dtype = np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[ ]:


data_absent.dtypes


# In[ ]:


##Chi square test
#Save categorical variable
cat_names = ["Reason for absence","Month of absence","Day of the week","Seasons","Work load Average/day ","Hit target","Disciplinary failure","Education","Son","Social drinker","Social smoker","Pet"]


# In[ ]:


cat_names


# In[ ]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data_absent['Absenteeism time in hours'],data_absent[i]))
    print(p)


# In[ ]:


#Drop the variables from data
data_absent = data_absent.drop(['ID','Weight','Day of the week','Education','Social smoker','Social drinker','Pet'], axis=1)


# In[ ]:


data_absent = data_absent.drop(['Work load Average/day'])


# In[ ]:


data_absent.head(3)


# In[ ]:


del data_absent['Work load Average/day']


# In[ ]:


#Normalisation
for i in cnames:
    print(i)
    data_absent[i] = (data_absent[i] - min(data_absent[i]))/(max(data_absent[i]) -min(data_absent[i]))


# In[ ]:


##############decision tree for regression############

#Divide data into train and test
train , test = train_test_split(data_absent , test_size = 0.2)


# In[ ]:


#decision tree for regresion
fit_DT = DecisionTreeRegressor(max_depth = 2).fit(train.iloc[:,0:12], train.iloc[:,12])


# In[ ]:


#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:12])


# In[ ]:


#Calculate MAPE
def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return mape


# In[ ]:


MAPE(test.iloc[:,9], predictions_DT)


# In[ ]:


######################## Regression Model ################


# In[ ]:


#Import libraries for LR
import statsmodels.api as sm

#Train the model using the training sets
model = sm.OLS(train.iloc[:,12],
              train.iloc[:,0:12]).fit()


# In[ ]:


#Print out the summary
model.summary()


# In[ ]:


#make the predictions by the model
predictions_LR = model.predict(test.iloc[:,0:12])


# In[ ]:


#Calculate MAPE
MAPE(test.iloc[:,12], predictions_LR)

