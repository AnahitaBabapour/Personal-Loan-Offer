#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import warnings


# In[127]:


data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
data


# In[128]:


data.describe()


# In[129]:


#in data describe we had minus num in min of Experince column. this is not true. we first count number of minus numbers and then replace them with zero.
data['Experience'][data['Experience']<0].count()


# In[130]:


#data['Experience'][data['Experience']<0] = 0
#may be user enter minous num in wrong way, we can create the posetive of that num.
for index, row in data.iterrows():
    if row['Experience'] < 0:
        data.at[index, 'Experience'] = -row['Experience']
data


# In[131]:


data['Experience'][data['Experience']<0].count()


# In[132]:


#sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
sns.lmplot(data = data, x = 'Income', y = 'Experience', hue = 'Personal Loan', palette = "pastel")
#we saw people with income higher than 50 may be accept loan!


# In[133]:


sns.jointplot(data = data, x = 'Family', y = 'Income', hue = 'Personal Loan', palette = 'Set2')


# In[134]:


sns.jointplot(data=data, x="Age", y="Income", hue="Personal Loan",palette = 'Set2')


# In[135]:


df = data.drop("ID",axis=1)
df = df.drop("ZIP Code",axis=1)
sns.pairplot(data=df, hue="Personal Loan",palette = 'Set2')


# In[136]:


df.corr()


# In[137]:


sns.lmplot(data=data, x="Age", y="CCAvg", hue="Personal Loan",palette = 'Set1')


# In[138]:


plt.scatter(x=data['CCAvg'],y=data['Income'])
plt.xlabel('CCAvg')
plt.ylabel('Income')


# In[139]:


plt.scatter(x=data['Income'],y=data['Personal Loan'])
plt.xlabel('Income')
plt.ylabel('Personal Loan')


# In[140]:


sns.relplot(data=data, x="Age", y="CCAvg", hue="Personal Loan")


# In[ ]:


#normalize
#scaler = MinMaxScaler()
#scaler.fit(data)
#x_scaled = scaler.transform(data)
#data = pd.DataFrame(x_scaled)
#print(data)


# In[141]:


x = df.drop("Personal Loan",axis=1)
y = df[["Personal Loan"]]


# In[142]:


logreg = LogisticRegression(solver='liblinear')
kfold = KFold(5)
results = cross_val_score(logreg, x, y, cv=kfold)
results


# In[143]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)


# In[144]:


print('Accuracy score: ', metrics.accuracy_score(y_test,y_pred))


# In[145]:


print(metrics.classification_report(y_test,y_pred))


# In[146]:


cm = metrics.confusion_matrix(y_pred,y_test)
print(cm)


# In[147]:


sns.heatmap(cm, annot = True, xticklabels = ['accept loan', 'reject loan'], yticklabels = ['accept loan', 'reject loan'])


# In[148]:


fpr,tpr,_ = metrics.roc_curve(y_test,y_pred)
plt.plot(fpr,tpr, label='data')
plt.legend(loc=4)
plt.show()


# In[149]:


y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label='data , AUC: '+ str(auc))
plt.legend(loc=4)


# In[150]:


print(logreg.intercept_)
print(logreg.coef_)


# In[151]:


multinb = MultinomialNB()
multinb.fit(x_train, y_train)
y_pred = multinb.predict(x_test)
y_pred


# In[152]:


print(metrics.classification_report(y_test,y_pred))


# In[153]:


cmm = metrics.confusion_matrix(y_test, y_pred)
print(cmm)


# In[154]:


sns.heatmap(cmm, annot= True, xticklabels=["accept loan","reject loan"], yticklabels=["accept loan","reject loan"])
plt.show()


# In[155]:


print(metrics.accuracy_score(y_test,y_pred))


# In[156]:


gsnb = GaussianNB()
gsnb.fit(x_train,y_train)
y_pr = gsnb.predict(x_test)
y_pr


# In[157]:


print(metrics.accuracy_score(y_test, y_pr))


# In[158]:


cm2 = metrics.confusion_matrix(y_test, y_pr) 
print(cm2)


# In[159]:


sns.heatmap(cm2, annot=True, xticklabels=["accept loan","reject loan"], yticklabels=["accept loan","reject loan"])
plt.show()


# In[160]:


print(metrics.classification_report(y_test,y_pr))


# In[161]:


cmnb = ComplementNB()
cmnb.fit(x_train, y_train)
y_pre = cmnb.predict(x_test)
print(metrics.classification_report(y_test, y_pre))


# In[162]:


cm3 = metrics.confusion_matrix(y_test, y_pre)
sns.heatmap(cm3, annot=True, xticklabels=['accept loan','reject loan'], yticklabels=['accept loan','reject loan'])


# In[163]:


for i in range(1,20):
    model = KNeighborsClassifier(i)
    model.fit(x_train, y_train)
    y_p = model.predict(x_test)
    print('accuracy score: ',metrics.accuracy_score(y_test, y_p))


# In[164]:


z =[]
for i in range(1,20):
    model = KNeighborsClassifier(i)
    model.fit(x_train, y_train)
    y_p = model.predict(x_test)
    z.append(metrics.accuracy_score(y_test, y_p))
print(z)


# In[165]:


cm4 = metrics.confusion_matrix(y_test, y_p)
print(cm4)


# In[166]:


sns.heatmap(cm4, annot= True, xticklabels=["accept loan","reject loan"], yticklabels=["accept loan","reject loan"])
plt.show()


# In[167]:


print(metrics.classification_report(y_test, y_p))


# In[168]:


train = []
test = []
k = range(1,21)
for i in k:
    model = KNeighborsClassifier(i)
    model.fit(x_train, y_train)
    train.append(model.score(x_train, y_train))
    test.append(model.score(x_test, y_test))
plt.plot(k, train)
plt.plot(k, test)
plt.show()


# In[169]:


parameters = {'n_neighbors' : range(1,21)}
gridsearch = GridSearchCV(estimator = model, param_grid = parameters, scoring = 'accuracy', cv = 5, verbose = 1, n_jobs = -1)
gridsearch.fit(x_train, y_train)


# In[170]:


gridsearch.best_params_


# In[171]:


model = KNeighborsClassifier(13)
model.fit(x_train, y_train)
y_p = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_p))

