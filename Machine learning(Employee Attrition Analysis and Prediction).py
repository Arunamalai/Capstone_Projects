%pip install pandas
import pandas as pd
data=pd.read_csv(r"C:\Users\Arunamalai\Downloads\Employee-Attrition - Employee-Attrition.csv")
data
data.info()
data.head()
data.describe()
data.isnull().sum()
contains_hyphen = data.applymap(lambda x: '-' in str(x))

print("Does the DataFrame contain the hyphen (-)?")
print(contains_hyphen)
contains_negative = data.select_dtypes(include=['number']).lt(0)
print(contains_negative) #to check negative value present in the dataframe
data[data.select_dtypes(include=['number']).lt(0)] = 0#replace negative value to 0
a=data.groupby('Attrition')
print(a.ngroups)
print(a.size())
a=data.groupby('Attrition').agg({'MonthlyIncome':'mean'})
a.apply(display)
b=data.groupby('Attrition').agg({'JobSatisfaction': 'mean'})
b.apply(display)
average_income = data['MonthlyIncome'].mean()
print(average_income)
min_income = data['MonthlyIncome'].min()
print(min_income)

max_income = data['MonthlyIncome'].max()
print(max_income)
filtered_data = data[(data['Attrition'] == 'Yes') & (data['MonthlyIncome'] > 6502)]
print(filtered_data)
data = data.applymap(lambda x: x.rstrip() if isinstance(x, str) else x)#Removing trailing space in string columns for full dataframe
print(data)
data[data.duplicated()]
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
correlation_matrix = data.select_dtypes(include='number').corr()


plt.figure(figsize=(6, 5)) # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

from statsmodels.stats.weightstats import ztest as ztest
import scipy.stats as stats
data1=data[data['Gender']=='Male']['MonthlyIncome']
data2=data[data['Gender']=='Female']['MonthlyIncome']
print(data1)
print(data2)

print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#independent

import scipy.stats as st
a=pd.crosstab(data['Gender'],data['MonthlyIncome'])
print(st.chi2_contingency(a))#Independent

a=pd.crosstab(data['Age'],data['Attrition'])
print(st.chi2_contingency(a))#dependent

data1=data[data['Attrition']=='Yes']['Age']
data2=data[data['Attrition']=='No']['Age']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data[data['Attrition']=='Yes']['JobSatisfaction']
data2=data[data['Attrition']=='No']['JobSatisfaction']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data[data['Attrition']=='Yes']['PerformanceRating']
data2=data[data['Attrition']=='No']['PerformanceRating']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#independent

data1=data[data['Attrition']=='Yes']['PercentSalaryHike']
data2=data[data['Attrition']=='No']['PercentSalaryHike']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#independent

data1=data[data['Attrition']=='Yes']['WorkLifeBalance']
data2=data[data['Attrition']=='No']['WorkLifeBalance']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data[data['Attrition']=='Yes']['Education']
data2=data[data['Attrition']=='No']['Education']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#Independent

a=data.groupby('PerformanceRating')
print(a.ngroups)
print(a.size())

#Annova
data1=data[data['PerformanceRating']==3]['Education']
print(data1)
data2=data[data['PerformanceRating']==4]['Education']
print(data2)

print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#Independent

data1=data[data['Attrition']=='Yes']['TotalWorkingYears']
data2=data[data['Attrition']=='No']['TotalWorkingYears']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True)) #dependent


data1=data['PerformanceRating']
data2=data['JobInvolvement']

print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data['PerformanceRating']
data2=data['JobSatisfaction']

print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data['PerformanceRating']
data2=data['Age']

print(ztest(data1,data2,value=0))
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#dependent

data1=data[data['Gender']=='Male']['PerformanceRating']
data2=data[data['Gender']=='Female']['PerformanceRating']
print(stats.ttest_ind(a=data1, b=data2, equal_var=True))#independent

#Feature Engineering
#Calculate OvertimeStress
data['OvertimeStress'] = data['OverTime'].map({'Yes': 1, 'No': 0}) * (data['DistanceFromHome'] + data['DailyRate'])

def tenure_category(years):
    if years < 2:
        return 'New'
    elif years < 5:
        return 'Junior'
    elif years < 10:
        return 'Mid'
    else:
        return 'Senior'

data['TenureCategory'] = data['YearsAtCompany'].apply(tenure_category)

data = data.drop(columns=['EmployeeNumber'])

data = data.drop(columns=['StandardHours'])

data = data.drop(columns=['Education'])

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
correlation_matrix = data.select_dtypes(include='number').corr()


plt.figure(figsize=(15, 12)) # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

#Preprocessing
print(data['MonthlyIncome'].describe())
data['MonthlyIncome'].skew() 

#Label Encoding(frequently used encoding)# to covert categoriacl to numerical

import pandas as pd
from sklearn.preprocessing import LabelEncoder



# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'Gender' column
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Apply label encoding to the 'Payment_Method' column
data['BusinessTravel'] = label_encoder.fit_transform(data['BusinessTravel'])

# Apply label encoding to the 'Payment_Method' column
data['JobRole'] = label_encoder.fit_transform(data['JobRole'])

data['OverTime'] = label_encoder.fit_transform(data['OverTime'])

data['MaritalStatus'] = label_encoder.fit_transform(data['MaritalStatus'])

data['Department'] = label_encoder.fit_transform(data['Department'])

data['TenureCategory'] = label_encoder.fit_transform(data['TenureCategory'])
data['EducationField'] = label_encoder.fit_transform(data['EducationField'])
data['Over18'] = label_encoder.fit_transform(data['Over18'])
#data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Show the DataFrame after encoding
data

a=data.groupby('Attrition')
print(a.ngroups)
print(a.size())

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Histogram with KDE (Kernel Density Estimate)
sns.histplot(data['MonthlyIncome'], kde=True, bins=30)
plt.title('Distribution of Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Frequency')
plt.show()


import numpy as np #fix the right scewed data using log transformation)

data['MonthlyIncome_log'] = np.log1p(data['MonthlyIncome'])  # log(1 + x) to handle 0s safely

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Histogram with KDE (Kernel Density Estimate)
sns.histplot(data['JobSatisfaction'], kde=True, bins=30)
plt.title('Distribution of JobSatisfaction')
plt.xlabel('JobSatisfaction')
plt.ylabel('Frequency')
plt.show()

data['JobSatisfaction'].skew() 

data['Age'] = np.log1p(data['Age'])

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Histogram with KDE (Kernel Density Estimate)
sns.histplot(data['MonthlyIncome_log'], kde=True, bins=30)
plt.title('Distribution of Monthly Income')
plt.xlabel('MonthlyIncome_log')
plt.ylabel('Frequency')
plt.show()

data['MonthlyIncome_log'].skew() 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

# Assuming 'df' is already defined and contains the necessary columns
#x = data[['Age','JobSatisfaction', 'BusinessTravel','WorkLifeBalance','TotalWorkingYears','OvertimeStress','EnvironmentSatisfaction','OverTime','JobInvolvement','JobLevel','MonthlyIncome','PercentSalaryHike','PerformanceRating','YearsWithCurrManager']]  # Feature columns
x = data[['Age','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','OverTime','JobInvolvement','MonthlyIncome_log','PercentSalaryHike','TenureCategory']]  # Feature columns
#x=data.drop("Attrition",axis=1)

y = data['Attrition']  # Target column

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

#Oversampling(smote technique)
from imblearn.over_sampling import SMOTE


smote = SMOTE()  # Create an instance of SMOTE


x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)  # Apply SMOTE


# You can check the class distribution after applying SMOTE
ax = y_train_smote.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling using SMOTE")

from imblearn.under_sampling import RandomUnderSampler#Rose(randomly delete data for balancing)


rus = RandomUnderSampler() # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String
x_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)


ax = y_train_rus.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")

scaler = MinMaxScaler()# to change max income as 5 min income as 1
x_train = scaler.fit_transform(x_train)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
x_test= scaler.transform(x_test)#test data=transform takes trainging data reference for its reference(that why add onlu transform not fit_transform. if we give fit_transfor it takes new min max values from test data) 
#(transform(old min max value(x_train)))

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Print classification report
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Write a code to save and load pickle file of the model
import pickle
#save the model to a pickle file
filename='logistic_regression_model.pkl'
pickle.dump(model,open(filename,'wb'))# this model containd trained mode(which is final value of y=mx+c)

#Load the model from the pickle file
loaded_model=pickle.load(open(filename,'rb'))

y_train_rus.value_counts()#check data balanced or not#output is balanced

x_train_smote.value_counts()

scaler = MinMaxScaler()# to change max income as 5 min income as 1
x_train_rus = scaler.fit_transform(x_train_rus)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
x_test= scaler.transform(x_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(x_train_rus, y_train_rus)

# Make predictions
y_pred = model.predict(x_test)

# Print classification report
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

y_train.value_counts()

#model evaluation
import pandas as pd

# Assuming y_pred and y_test are already defined
# Create a crosstab to compare the predicted vs actual values
crosstab = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

# Print the crosstab
print(crosstab)

#prediction from user input
df=pd.DataFrame({'Age':[49],'JobSatisfaction':[2],'EnvironmentSatisfaction':[3],'WorkLifeBalance':[3],'OverTime':[0],'JobInvolvement':[2],'MonthlyIncome_log':[5130],'PercentSalaryHike':[23],'TenureCategory':[3]})#()for scaling we can create data frame format
a=scaler.transform(df)# minmax scaling process for given predicted values
model.predict(a)


#DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier
x = data[['Age','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','OverTime','JobInvolvement','MonthlyIncome','PercentSalaryHike','TenureCategory']]  # Feature columns
y=data['Attrition']
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
x_train_ros = scaler.fit_transform(x_train_ros)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
x_test= scaler.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train_ros,y_train_ros)
y_pred=dt.predict(x_test)

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
    scores = cross_val_score(tree, x, y, cv=5, scoring='accuracy')  # 5-fold CV
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
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_ros, y_train_ros)
# Get the best parameters
print("Best parameters from Grid Search CV:", grid_search.best_params_)
print("Best accuracy from Grid Search CV:", grid_search.best_score_)
# Train with best parameters
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(x_test)
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
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(x_train_ros, y_train_ros)
# Get the best parameters
print("Best parameters from Grid Search CV:", random_search.best_params_)
print("Best accuracy from Grid Search CV:", random_search.best_score_)
# Train with best parameters
best_model_grid = random_search.best_estimator_
y_pred_grid = best_model_grid.predict(x_test)
# Performance Metrics
print("\nGrid Search CV Classification Report:\n", classification_report(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred))


#Random Forest
from sklearn.model_selection import train_test_split
#x=data.drop("Attrition",axis=1)
x = data[['Age','JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance','OverTime','JobInvolvement','MonthlyIncome_log','PercentSalaryHike','TenureCategory']]  # Feature columns

y = data['Attrition']  # Target column
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

smote = SMOTE()  # Create an instance of SMOTE


x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)  # Apply SMOTE



scaler = MinMaxScaler()# to change max income as 5 min income as 1
x_train_smote = scaler.fit_transform(x_train_smote)#scaler object contains fit_transform= min max values as their reference(fresh min max values)
x_test= scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
dt=RandomForestClassifier(n_estimators=10)
dt.fit(x_train_smote,y_train_smote)
y_pred=dt.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#Support Vector algorithm
from  sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model=SVC()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)

print(models)

#create a dataframe with y_test and y_pred
import pandas as pd
pd.set_option('display.max_rows', None)
df=pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
print(df) 


import pandas as pd

# Set pandas option to display all rows
pd.set_option('display.max_rows', None)

# Example DataFrame
df = pd.read_csv('your_file.csv')  # or however you create your DataFrame

print(df) 

import pickle
#save the model to a pickle file
filename='logistic_regression_model.pkl'
pickle.dump(model,open(filename,'wb'))# this model containd trained mode(which is final value of y=mx+c)

#Load the model from the pickle file
loaded_model=pickle.load(open(filename,'rb'))

loaded_model.predict(a)

print(type(model)) 


import streamlit as st
import pickle
import numpy as np

# Load the trained model
loaded_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# App Configuration
st.set_page_config(page_title="Attrition Prediction", page_icon="🔍", layout="wide")
st.markdown(
    "<h1 style='color: #3E64FF; text-align: center;'>Employee Attrition Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("### 🎯 Predict whether an employee is likely to leave the company based on HR metrics.")

# Input Form
with st.form("prediction_form"):
    st.subheader("📝 Input Employee Information")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        JobSatisfaction = st.selectbox("Job Satisfaction", 
                                       ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"])
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", 
                                               ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"])
        WorkLifeBalance = st.selectbox("Work-Life Balance", 
                                       ["Bad (1)", "Good (2)", "Better (3)", "Best (4)"])
        OverTime = st.radio("OverTime", ["No", "Yes"])

    with col2:
        JobInvolvement = st.selectbox("Job Involvement", 
                                      ["Low (1)", "Medium (2)", "High (3)", "Very High (4)"])
        MonthlyIncome_log = st.number_input("Monthly Income (Log Scaled)", min_value=0.0, step=0.1)
        PercentSalaryHike = st.slider("Percent Salary Hike", 0, 100, 15)
        TenureCategory_label = st.selectbox(
            "Tenure Category (Years of Experience)",
            ["New (<2 years)", "Junior (2-4 years)", "Senior (5-9 years)", "Super Senior (10+ years)"]
        )

    submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        # Convert categorical text to numeric values
        JobSatisfaction = int(JobSatisfaction[JobSatisfaction.find('(')+1])
        EnvironmentSatisfaction = int(EnvironmentSatisfaction[EnvironmentSatisfaction.find('(')+1])
        WorkLifeBalance = int(WorkLifeBalance[WorkLifeBalance.find('(')+1])
        JobInvolvement = int(JobInvolvement[JobInvolvement.find('(')+1])
        OverTime_encoded = 1 if OverTime == "Yes" else 0

        # Map Tenure Category to numeric value
        def map_tenure_label_to_numeric(label):
            if "New" in label:
                return 1
            elif "Junior" in label:
                return 3
            elif "Senior" in label:
                return 7
            else:  # Super Senior
                return 11

        TenureCategory_numeric = map_tenure_label_to_numeric(TenureCategory_label)

        # Create feature vector with numeric value for TenureCategory
        input_data = np.array([[Age, JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, 
                                OverTime_encoded, JobInvolvement, MonthlyIncome_log, 
                                PercentSalaryHike, TenureCategory_numeric]])

        # Prediction
        prediction = loaded_model.predict(input_data)
        probability = loaded_model.predict_proba(input_data)

        # Attrition labels
        attrition_labels = {"No": "No Attrition", "Yes": "Yes Attrition"}
        predicted_label = attrition_labels[prediction[0]]

        # Results
        st.markdown("---")
        st.markdown(f"<h3 style='color: #28A745;'>✅ Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>📊 Confidence: {np.max(probability)*100:.2f}%</h4>", unsafe_allow_html=True)
        st.write(f"📌 Experience Level: **{TenureCategory_label}**")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Model Accuracy: 86% | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

