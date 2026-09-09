# how to predict the charges
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('insurance.csv') # to load th data
print(data)

print(data.describe()) # give all the data are mean,mid,mode,std etc find.

print(data.head()) # first 5 value

print(data.tail())  # last 5 value on a given data.

print(data.info())

print(data.isnull().sum()) # give the all null value.(EDA used.)

print(data.columns)  # to check the column names '

numeric_columns = ['age', 'bmi', 'children', 'charges'] # 4 graph plot

for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show() # kde means define how to ploat a curve.

sns.countplot(x=data['children'])
plt.show()

sns.countplot(x=data['sex'])
plt.show()

sns.countplot(x=data['smoker'])   # data vizulations 
plt.show()

for col in numeric_columns:
    plt.figure(figsize=(6,4)) # input and output both are included.
    sns.boxplot(x=data[col])
    plt.show()

# plot a correlation (best used to find a heatmap)
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True),annot=True) # it is all part of EDA.
plt.show() 

# Data Cleaning and preprocessing.

data_cleaned = data.copy()

print("Original rows:", len(data_cleaned))

print(data_cleaned.shape) # look at the without clean the shape

data_cleaned.drop_duplicates(inplace=True)

print("Rows after removing duplicates:", len(data_cleaned))
print(data_cleaned.head())

print(data_cleaned.shape)  # clean and then look at the shape

print(data_cleaned.isnull().sum()) # it will be store a null value.

print(data_cleaned.dtypes) # data type are define what data type are stored.

print(data_cleaned['sex'].value_counts()) # it will be count the value of male or female.
# it is a level encoding.


data_cleaned['sex']=data_cleaned ['sex'].map({'male':0,'female':1})
print(data_cleaned.head())

print(data_cleaned['smoker'].value_counts())  # check the smokers data

data_cleaned['smoker']=data_cleaned ['smoker'].map({'no':0,'yes':1})
print(data_cleaned)     # level encoading used 

data_cleaned.rename(columns={
    'sex':'is_female',
    'smoker':'is_smoker'
},inplace=True)
print(data_cleaned.head())

print(data['region'].value_counts()) # count the value of regions

# used on-hot encoding
data_cleaned=pd.get_dummies(data_cleaned,columns=['region'],drop_first=True)
print(data_cleaned.head())

# convert all 0 and 1 form 
data_cleaned = data_cleaned.astype(int)
print(data_cleaned)

# Feature engineering and Extraction

sns.histplot(data['bmi'])
plt.show()

# import the pandas of the given data
data_cleaned['bmi_category']=pd.cut(
    data_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['underweight','normal','overweight','obese']
)
print(data_cleaned)

data_cleaned=pd.get_dummies(data_cleaned,columns=['bmi_category'],drop_first=True)
print(data_cleaned.head())

data_cleaned=data_cleaned.astype(int)
print(data_cleaned.head())

# feature scaling

print(data_cleaned.columns)

from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler = StandardScaler()
data_cleaned[cols]=scaler.fit_transform(data_cleaned[cols])
print(data_cleaned[cols].head())

# Feature Extraction 


#pearson correlation calculation 
#feature to check against target.
from scipy.stats import pearsonr

selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_normal', 'bmi_category_overweight',
    'bmi_category_obese'
]

correlations = {
    feature: pearsonr(data_cleaned[feature], data_cleaned['charges'])[0]
    for feature in selected_features
}

correlations_data = pd.DataFrame(list(correlations.items()),columns=['Feature', 'Pearson Correlation'])

correlations_data = correlations_data.sort_values(by='Pearson Correlation', ascending=False)

print("\n--- Pearson Correlation Results ---")
print(correlations_data)

import pandas as pd
from scipy.stats import chi2_contingency

# Categorical features
cat_features = [
    'is_female',
    'is_smoker',
    'region_northwest',
    'region_southeast',
    'region_southwest',
    'bmi_category_normal',
    'bmi_category_overweight',
    'bmi_category_obese'
]

# Significance level
alpha = 0.05

# Divide charges into 4 groups
data_cleaned['charges_bin'] = pd.qcut(
    data_cleaned['charges'],
    q=4,
    labels=False
)

# Store Chi-Square results
chi2_results = {}

for col in cat_features:

    # Create contingency table
    contingency = pd.crosstab(
        data_cleaned[col],
        data_cleaned['charges_bin']
    )

    # Chi-Square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency)

    # Decision
    if p_val < alpha:
        decision = 'Reject Null (Keep Feature)'
    else:
        decision = 'Accept Null (Drop Feature)'

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

# Convert dictionary to DataFrame
chi2_df = pd.DataFrame(chi2_results).T

# Sort by p-value
chi2_df = chi2_df.sort_values(by='p_value')

# Print result
print("\nChi-Square Test Results:")
print(chi2_df.to_string())

final_df = data_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker','charges', 'region_southeast', 'bmi_category_obese']]

print(final_df)

# train_test_split
from sklearn.model_selection import train_test_split
X = final_df.drop('charges', axis=1)
y = final_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# used by a linear regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", y_pred[:5])



from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("R2 Score:", r2)
print("Adjusted R2:", adjusted_r2)