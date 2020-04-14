#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
import gc
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import randint
seed = 330
np.random.seed(seed)


# In[2]:


pwd


# In[3]:


cd /Users/mattpucci/Desktop


# In[4]:


ls


# # Let's take an exploratory look at the NBA 2016-2017 team box score dataset and do some Preprocessing. 

# In[108]:


df= pd.read_csv('2016-17_teamBoxScore.csv')
df.head()


# In[137]:


date_value = pd.to_datetime(df['gmDate'], errors='coerce')
time_value = pd.to_datetime(df['gmTime'], errors='coerce')


df['year'] = date_value.dt.year 
df['month'] = date_value.dt.month 
df['day'] = date_value.dt.day 
df['hour'] = time_value.dt.hour 
df['minute'] = time_value.dt.minute

del df['gmDate']
del df['gmTime']


# In[139]:


cols_with_missing = [col for col in df.columns if df[col].isnull().any()] 
df.drop(cols_with_missing, axis=1, inplace=True)
df_test.drop(cols_with_missing, axis=1, inplace=True)


# In[140]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = df.columns.values.tolist()
for col in features:
    if df[col].dtype in numerics: continue
    categorical_columns.append(col)
indexer = {}
for col in categorical_columns:
    if df[col].dtype in numerics: continue
    _, indexer[col] = pd.factorize(df[col])
    
for col in categorical_columns:
    if df[col].dtype in numerics: continue
    df[col] = indexer[col].get_indexer(df[col])


# In[141]:


df.info()


# In[116]:


df.info()


# In[142]:


df.hist(figsize=(30,30))


# ## Not all of the data looks standardly distributed.  Let's now check and see if there are any strong correlations.

# In[143]:


df_corr = df.corr()
f, ax = plt.subplots(figsize=(25,25))
sns.heatmap(df_corr, vmax=.8, square=True).set_title('Figure 1: Seaborn Data Correlation Heat Map') 


# ### There is far too much going on in the above visual to make good sense of it.  Let's take a look at the 10 highest correlated features with more clarity.  

# In[144]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[145]:


df.teamRslt.value_counts()


# In[146]:


df.teamRslt=df.teamRslt.map(lambda x: '0' if x=='Loss' else x)


# In[147]:


df.teamRslt=df.teamRslt.map(lambda x: '1' if x=='Win' else x)


# In[148]:


#Above we just converted a Loss to a value of 0, with a win a value of 1.


# In[149]:


k = 12
cols = df_corr.nlargest(k, 'teamPTS')['teamPTS'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
teampts_heat = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("Figure 2: Concentrated Heat Map with Interest Variable = Team Points ")
plt.figure(figsize=(25,25))
plt.show()


# In[150]:


k = 12
cols = df_corr.nlargest(k, 'teamDrtg')['teamDrtg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
teampts_heat = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("Figure 2: Concentrated Heat Map with Interest Variable = Team Defensive Rating ")
plt.figure(figsize=(25,25))
plt.show()


# In[151]:


k = 12
cols = df_corr.nlargest(k, 'teamOrtg')['teamOrtg'].index
f, ax = plt.subplots(figsize=(10,6))
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
teampts_heat = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("Figure 2: Concentrated Heat Map with Interest Variable = Team Offensive Rating ")
plt.figure(figsize=(25,25))
plt.show()


# In[152]:


df.info()


# In[153]:


import statsmodels.api as sm  #statsmodels forward regression used to determine which features are the best fit based on P-Value.
import pandas as pd
import numpy as np
def forward_regression(X, y,
                       threshold_in=0.01,
                       verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
        if not changed:
            break
    return included


# In[154]:


X = df
y = df.teamRslt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=47)
i = 1
results_df = pd.DataFrame()
predictors = list()
reg_score = list()
mse_diffs = list()
added_pred = list()
previous_columns = []


# In[155]:


forward_regression(X,y)


# In[156]:


feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA','teamFGM']
x = df[feature_cols]
y = df['teamRslt']
x.head()


# In[157]:


df_test = pd.read_csv('2017-18_teamBoxScore.csv')


# In[158]:


x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.4, random_state=2)


# In[159]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[161]:


train_set = lgbm.Dataset(X_train, y_train, silent=False)
valid_set = lgbm.Dataset(X_valid, y_valid, silent=False)

params = {
        'boosting_type':'gbdt', 'objective': 'regression', 'num_leaves': 31,
        'learning_rate': 0.05, 'max_depth': -1, 'subsample': 0.8,
        'bagging_fraction' : 1, 'max_bin' : 5000 , 'bagging_freq': 20,
        'colsample_bytree': 0.6, 'metric': 'rmse', 'min_split_gain': 0.5,
        'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight':1,
        'zero_as_missing': True, 'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)

fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();


# In[160]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, pred))
print(knn.predict_proba(x_test))


# In[135]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='auto', max_iter=101)
model.fit(x_train, y_train)
print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))


# In[38]:


class_labels = list(df['Class_Type'])
labels = [1,2,3,4,5,6,7]


# In[101]:


top_stats = df
target = df.teamRslt
target.columns = ['teamRslt']


# In[102]:


clf = LinearSVC(random_state=2)
clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)
pred = (clf.predict(x_test))
#print(pred)
print(metrics.accuracy_score(y_test, pred))


# In[103]:


clf = RandomForestClassifier()
clf.fit(x_train, y_train)


# In[104]:


print(clf.feature_importances_)


# In[107]:


pred = clf.predict(x_test)
print(metrics.accuracy_score(y_test, pred))


# In[246]:


clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
clfgtb.score(x_test, y_test)


# In[ ]:





# In[ ]:





# In[225]:


df.teamAbbr = df.teamAbbr.astype('category')
df.teamConf = df.teamConf.astype('category')
df.teamDiv = df.teamDiv.astype('category')
df.teamLoc = df.teamLoc.astype('category')

# Now that our features are category variables, lets assign some dummie data to represent them.
teamAbbr = pd.get_dummies(df.teamAbbr, prefix='Team Abbreviation', drop_first=True)
teamConf = pd.get_dummies(df.teamConf, prefix='grade', drop_first=True)
teamDiv = pd.get_dummies(df.teamDiv, prefix='zipcode', drop_first=True)
teamLoc = pd.get_dummies(df.teamLoc, prefix='bedrooms', drop_first=True)

# Add our transformed variables to the data set and remove the originals.
df = df.join([teamAbbr, teamConf, teamDiv, teamLoc])


# In[227]:


df.info()


# In[230]:


X_1, y_1 = df[[x for x in df.columns if x != 'teamRslt']], df[['teamRslt']]
X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size = 0.2, random_state=123)

X_2, y_2 = df1[[x for x in df1.columns if x != 'teamRslt']], df1[['teamRslt']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, y_2, test_size = 0.2, random_state=123)


# In[238]:


print(X_train.shape)
print(y_train.shape)


# In[235]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test

mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test =np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)


# In[114]:


plt.figure(figsize=(20,15))

plt.subplot(221)
sns.lineplot(df.teamOrtg, df.teamPTS)
plt.title('Effect of Offensive Rating on Points')
plt.subplot(222)
sns.lineplot(df.teamDrtg, df.teamPTS)
plt.title('Effect of Defensive Rating on Points')
plt.subplot(223)
sns.lineplot(df.teamFIC, df.teamPTS)
plt.title('Effect of Waterfront on Price')
plt.subplot(224)
sns.lineplot(df.teamFGM, df.teamPTS)
plt.title('Effect of House Condition on Price')
sns.lineplot(df.opptDrtg, df.teamPTS)
plt.title('Effect of House Condition on Price')
sns.lineplot(df.opptOrtg, df.teamPTS)
plt.title('Effect of House Condition on Price')

plt.subplots_adjust(hspace=0.40)
sns.set_style('darkgrid')
sns.color_palette('pastel')
plt.show()


# In[113]:


df('opptDayOff').value_count()


# In[247]:


pwd


# In[ ]:




