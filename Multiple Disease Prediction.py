%pip install pandas
#Project Title:Multiple disease prediction (ML Project)
import pandas as pd
data=pd.read_csv(r"C:\Users\Arunamalai\Downloads\parkinsons - parkinsons.csv")
data

data.info()
data.head()
data.describe()
data.isnull().sum()
a=data.groupby('status')
print(a.ngroups)
print(a.size())

b=data.groupby('status').agg({'MDVP:Fo(Hz)':'mean'})
print(b)

c=data.groupby('status').agg({'MDVP:Jitter(%)':'mean'})
print(c)

filtered_data=data[(data['status']==1) & (data['MDVP:Jitter(%)']<0.003866)]

result_data = filtered_data[['status', 'MDVP:Jitter(%)']]
print(result_data)

min_Jitter=data['MDVP:Jitter(%)'].min()
print(min_Jitter)
max_Jitter=data['MDVP:Jitter(%)'].max()
print(max_Jitter)
data=data.applymap(lambda x:x.strip() if isinstance(x,str) else x)
print(data)
data[data.duplicated()]

#Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix=data.select_dtypes(include='number').corr()

plt.figure(figsize=(15,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

#2.inferential Analysis/EDA 
#hypothesis testing(ztest,ttest)(one sample mean test - single variable(single column))
%pip install statsmodel

#Ztest
from statsmodels.stats.weightstats import ztest as ztest
import scipy.stats as stats
data1=data[data['status']==0]['MDVP:RAP']
data2=data[data['status']==1]['MDVP:RAP']

print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1,b=data2,equal_var=True))#dependent

data1=data[data['status']==0]['MDVP:Fo(Hz)']
data2=data[data['status']==1]['MDVP:Fo(Hz)']
print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1,b=data2,equal_var=True))#dependent

data1=data[data['status']==0]['MDVP:Fo(Hz)']
data2=data[data['status']==1]['MDVP:Fo(Hz)']
print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1,b=data2,equal_var=True))

data1=data['MDVP:Fhi(Hz)']
data2=data['MDVP:Fo(Hz)']
print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1,b=data2,equal_var=True))

data['MDVP:Fo(Hz)'].skew()

#Histogram plot for checking skewness
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.histplot(data['MDVP:Fhi(Hz)'],kde=True,bins=30)
plt.title('Distribution of MDVP:Fhi(Hz)')
plt.xlabel('MDVP:Fhi(Hz)')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.histplot(data['MDVP:Jitter(%)'],kde=True,bins=30)
plt.title('Distribution of MDVP:Jitter(%)')
plt.xlabel('MDVP:Jitter(%)')
plt.ylabel('Frequency')
plt.show()   

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.histplot(data['MDVP:Jitter(Abs)'],kde=True,bins=30)
plt.title('Distribution of MDVP:Jitter(Abs)')
plt.xlabel('MDVP:Jitter(Abs)')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.histplot(data['MDVP:RAP'],kde=True,bins=30)
plt.title('Distribution of MDVP:RAP')
plt.xlabel('MDVP:RAP')
plt.ylabel('Frequency')
plt.show()

%pip install scikit-learn

#Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#feature selection
x=data.drop(columns=['status','name'])
#x=data[['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','Shimmer:DDA','NHR','HNR','RPDE','DFA']]
y=data['status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30) 	

%pip install imblearn

#OverSampling(SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_smote,y_train_smote=smote.fit_resample(x_train,y_train)
ax = y_train_smote.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling using SMOTE")

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(x_train)
X_test =  scaler.transform(x_test)

from sklearn.metrics import confusion_matrix
model= LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))# used with SMOTE, Without SMOTE and model= LogisticRegression(class_weight='balanced', random_state=42),while compare the output, I chose class_weight='balanced'

y.value_counts()

#model evaluation
import pandas as pd

# Assuming y_pred and y_test are already defined
# Create a crosstab to compare the predicted vs actual values
crosstab = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

# Print the crosstab
print(crosstab)

#prediction from userinput
df=pd.DataFrame({'MDVP:Fo(Hz)':[119.992],'MDVP:Fhi(Hz)':[157.302],'MDVP:Flo(Hz)':[74.997],'MDVP:Jitter(%)':[0.00784],'MDVP:Jitter(Abs)':[0.00007],'MDVP:RAP':[0.0037],'MDVP:PPQ':[0.00554],'Jitter:DDP':[0.01109],'MDVP:Shimmer':[0.04374],'MDVP:Shimmer(dB)':[0.426],'Shimmer:APQ3':[0.02182],'Shimmer:APQ5':[0.0313],'MDVP:APQ':[0.02971],'Shimmer:DDA':[0.06545],'NHR':[0.02211],'HNR':[21.033],'RPDE':0.414783,'DFA':[0.815285],'spread1':[-4.813031],'spread2':[0.266482],'D2':[2.301442],'PPE':[0.284654]})
a=scaler.transform(df)# minmax scaling process for given predicted values
model.predict(a)

#DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier
#x=data.drop(columns=['status','name'])
x=data[['MDVP:Jitter(%)','MDVP:Jitter(Abs)','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:DDA','NHR','HNR','RPDE','DFA']]
y=data['status']
from sklearn.model_selection import train_test_split
# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

from imblearn.over_sampling import RandomOverSampler


ros = RandomOverSampler(sampling_strategy=1) # Float
#ros = RandomOverSampler(sampling_strategy="not majority") # String
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)


ax = y_train_ros.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")

y_train_ros.value_counts()

scaler = MinMaxScaler()# to change max income as 5 min income as 1
X_train = scaler.fit_transform(x_train)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
X_test= scaler.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# You can vary tree depths or other hyperparameters if you want
max_depth_range = 5
mean_acc = np.zeros(max_depth_range)

for depth in range(1, max_depth_range + 1):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, x, y, cv=5, scoring='f1')  # 5-fold CV & f1 mean balances precision and recall
    mean_acc[depth - 1] = scores.mean()

print(mean_acc)

#feature importatce
import matplotlib.pyplot as plt
feature_importances = dt.feature_importances_

# Convert to DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

#grid search cv, random search cv

from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier()
# Define hyperparameter grid for tuning
param_grid = {
   'criterion': ['gini', 'entropy','log_loss'],
   'max_depth': [5, 10, 15, None],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4]
}
# 1. **Grid Search CV** (Exhaustive Search for Best Hyperparameters)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
# Get the best parameters
print("Best parameters from Grid Search CV:", grid_search.best_params_)
print("Best accuracy from Grid Search CV:", grid_search.best_score_)
# Train with best parameters
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
# Performance Metrics
print("\nGrid Search CV Classification Report:\n", classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import RandomizedSearchCV
model = DecisionTreeClassifier()
# Define hyperparameter grid for tuning
param_grid = {
   'criterion': ['gini', 'entropy','log_loss'],
   'max_depth': [5, 10, 15, None],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4]
}
# 1. **Grid Search CV** (Exhaustive Search for Best Hyperparameters)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='f1', n_jobs=-1)
random_search.fit(X_train, y_train)
# Get the best parameters
print("Best parameters from Grid Search CV:", random_search.best_params_)
print("Best accuracy from Grid Search CV:", random_search.best_score_)
# Train with best parameters
best_model_grid = random_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
# Performance Metrics
print("\nGrid Search CV Classification Report:\n", classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

model = DecisionTreeClassifier()

# Define hyperparameter grid for tuning
param_grid = {
   'criterion': ['gini', 'entropy', 'log_loss'],
   'max_depth': [5, 10, 15, None],
   'min_samples_split': [2, 5, 10],
   'min_samples_leaf': [1, 2, 4]
}

# 1. **Randomized Search CV** (Random Sampling for Hyperparameter Tuning)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42  # Optional for reproducibility
)

random_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters from Randomized Search CV:", random_search.best_params_)
print("Best accuracy from Randomized Search CV:", random_search.best_score_)

# Train with best parameters
best_model_grid = random_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)

# Performance Metrics
print("\nRandomized Search CV Classification Report:\n", classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))

#Random Forest
from sklearn.model_selection import train_test_split
#x=data.drop("Attrition",axis=1)
x=data[['MDVP:Jitter(%)','MDVP:Jitter(Abs)','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:DDA','NHR','HNR','RPDE','DFA']]
y=data['status']
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

#smote = SMOTE()  # Create an instance of SMOTE


#x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)  # Apply SMOTE



scaler = MinMaxScaler()# to change max income as 5 min income as 1
X_train = scaler.fit_transform(x_train)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
X_test= scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
dt=RandomForestClassifier(n_estimators=10,random_state=42,class_weight='balanced')
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Support Vector algorithm
from  sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model=SVC(class_weight='balanced',random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

%pip install lazypredict
%pip install ipywidgets

from tqdm import tqdm
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Feature matrix and target (assuming you already have these)
x=data[['MDVP:Jitter(%)','MDVP:Jitter(Abs)','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:DDA','NHR','HNR','RPDE','DFA']]
y=data['status']

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# (Optional) scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
