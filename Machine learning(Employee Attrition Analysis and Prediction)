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
