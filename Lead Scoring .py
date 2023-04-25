#!/usr/bin/env python
# coding: utf-8

# **Lead Scoring Case Study**
# *By: Sooraj Kushwaha*
# 
# **Problem Statement**
# 
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. There are a lot of leads generated in the initial stage but only a few of them come out as paying customers. The company needs to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc.) in order to get a higher lead conversion.
# 
# The problem is to help the comapany select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.
# 
# **Data**
# Leads.csv :
# The dataset consists of various attributes such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, etc. which may or may not be useful in ultimately deciding whether a lead will be converted or not. The target variable, in this case, is the column ‘Converted’ which tells whether a past lead was converted or not wherein 1 means it was converted and 0 means it wasn’t converted.

# In[87]:


# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# To suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[88]:


# Loading the data using Pandas
df_leads_original =pd.read_csv(r'C:\Users\USER\Desktop\Lead scoring\Leads.csv')
# To keep an original copy
df_leads = df_leads_original.copy()
df_leads.head()


# # 1.1 Inspect Data frame 

# In[89]:


# The .info() code gives almost the entire information that needs to be inspected, so let's start from there
df.info()


# In[90]:


#To get the idea of how the table looks like we can use .head() or .tail() command
df.head()


# In[15]:


# The .shape code gives the no. of rows and columns
df.shape


# In[16]:


#To get an idea of the numeric values, use .describe()
df.describe()


# # Cleaning 

# In[91]:


# To check for duplicates
df_leads.loc[df_leads.duplicated()]


# No dublicates
# 

# In[92]:


# To check for duplicates in columns
print(sum(df_leads.duplicated(subset = 'Lead Number')))
print(sum(df_leads.duplicated(subset = 'Prospect ID')))


# As the values in these columns are different for each entry/row, there are just indicative of the ID and are not important from an analysis point of view. Hence, can be dropped.

# In[93]:


df_leads = df_leads.drop(['Lead Number','Prospect ID'],1)


# As it can be seen, there are select values in many columns. This means that the person did not select any option for the given field. Hence, these are like NULL values.
# 
# 

# In[94]:


# To convert 'Select' values to NaN
df_leads = df_leads.replace('Select', np.nan)


# In[95]:


# To get percentage of null values in each column
round(100*(df_leads.isnull().sum()/len(df_leads.index)), 2)


# We'll drop columns with more than 50% of missing values as it does not make sense to impute these many values. But the variable 'Lead Quality', which has 51.6% missing values seems promising. So we'll keep it for now.

# In[96]:


# To drop columns with more than 50% of missing values as it does not make sense to impute these many values
df_leads = df_leads.drop(df_leads.loc[:,list(round(100*(df_leads.isnull().sum()/len(df_leads.index)), 2)>52)].columns, 1)


# For categorical variables, we'll analyse the count/percentage plots.
# For numerical variable, we'll describe the variable and analyse the box plots.

# In[97]:


# Function for percentage plots
def percent_plot(var):
    values = (df_leads[var].value_counts(normalize=True)*100)
    plt_p = values.plot.bar(color=sns.color_palette('deep'))
    plt_p.set(xlabel = var, ylabel = '% in dataset')


# In[98]:


# For Lead Quality
percent_plot('Lead Quality')


# Null values in the 'Lead Quality' column can be imputed with the value 'Not Sure' as we can assume that not filling in a column means the employee does not know or is not sure about the option.

# In[99]:


df_leads['Lead Quality'] = df_leads['Lead Quality'].replace(np.nan, 'Not Sure')


# In[100]:


# For 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
asym_list = ['Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score', 'Asymmetrique Profile Score']
plt.figure(figsize=(10, 7))
for var in asym_list:
    plt.subplot(2,2,asym_list.index(var)+1)
    if 'Index' in var:
        sns.countplot(df_leads[var])
    else:
        sns.boxplot(df_leads[var])
plt.show()


# In[101]:


# To describe numerical variables
df_leads[asym_list].describe()


# These four variables have more than 45% missing values and it can be seen from the plots that there is a lot of variation in them. So, it's not a good idea to impute 45% of the data. Even if we impute with mean/median for numerical variables, these values will not have any significant importance in the model. We'll have to drop these variables.

# In[102]:


df_leads = df_leads.drop(asym_list,1)


# In[103]:


# To get percentage of null values in each column
round(100*(df_leads.isnull().sum()/len(df_leads.index)), 2)


# In[104]:


# For 'City'
percent_plot('City')


# Around 60% of the City values are Mumbai. We can impute 'Mumbai' in the missing values.
# 
# 

# In[106]:


df_leads['City'] = df_leads['City'].replace(np.nan, 'Mumbai')
# For 'Specialization'
percent_plot('Specialization')


# There are a lot of different specializations and it's not accurate to directly impute with the mean. It is possible that the person does not have a specialization or his/her specialization is not in the options. We can create a new column for that.
# 

# In[107]:


df_leads['Specialization'] = df_leads['Specialization'].replace(np.nan, 'Others')


# In[108]:


# For 'Tags', 'What matters most to you in choosing a course', 'What is your current occupation' and 'Country'
var_list = ['Tags', 'What matters most to you in choosing a course', 'What is your current occupation', 'Country']

for var in var_list:
    percent_plot(var)
    plt.show()


# In[46]:


sns.countplot(df['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# In[109]:


# To impute with the most frequent value
for var in var_list:
    top_frequent = df_leads[var].describe()['top']
    df_leads[var] = df_leads[var].replace(np.nan, top_frequent)


# In[48]:


plt.figure(figsize = (10,10))
plt.subplot(221)
plt.hist(df_final['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(222)
plt.hist(df_final['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(223)
plt.hist(df_final['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# In[110]:


# To get percentage of null values in each column
round(100*(df_leads.isnull().sum()/len(df_leads.index)), 2)


# In[111]:


# For 'TotalVisits' and 'Page Views Per Visit'
visit_list = ['TotalVisits', 'Page Views Per Visit']
plt.figure(figsize=(15, 5))
for var in visit_list:
    plt.subplot(1,2,visit_list.index(var)+1)
    sns.boxplot(df_leads[var])
plt.show()

df_leads[visit_list].describe()


# From the above analysis, it can be seen that there is a lot of variation in both of the variables. As the percentage of missing values for both of them are less than 2%, it is better to drop the rows containing missing values.

# In[112]:


# For 'Lead Source' and 'Last Activity'
var_list = ['Lead Source', 'Last Activity']

for var in var_list:
    percent_plot(var)
    plt.show()


# In[52]:


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= df_final).tick_params(axis='x', rotation = 90)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= df_final).tick_params(axis='x', rotation = 90)
plt.title('What is your current occupation')
plt.show()


# In these categorical variables, imputing with the most frequent value is not accurate as the next most frequent value has similar frequency. Also, as these variables have very little missing values, it is better to drop the rows containing these missing values. Hence, we'll drop the rows containing any missing missing values for above four variables.

# In[113]:


# To drop the rows containing missing values
df_leads.dropna(inplace = True)


# In[114]:


# To get percentage of null values in each column
round(100*(df_leads.isnull().sum()/len(df_leads.index)), 2)


# There are no more missing values. Data is cleaned!
# 
# 
# 

# **Data Visualization**

# In[115]:


# For the target variable 'Converted'
percent_plot('Converted')


# In[116]:


(sum(df_leads['Converted'])/len(df_leads['Converted'].index))*100


# 37.8% of the 'Converted' data is 1 ie. 37.8% of the leads are converted. This means we have enough data of converted leads for modelling.
# 
# 

# **Visualising Numerical Variables and Outlier Treatment**
# 
# 
# 

# **3. Dummy Variables**

# In[118]:


# Boxplots
num_var = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']
plt.figure(figsize=(15, 10))
for var in num_var:
    plt.subplot(3,1,num_var.index(var)+1)
    sns.boxplot(df_leads[var])
plt.show()


# In[119]:


df_leads[num_var].describe([0.05,.25, .5, .75, .90, .95])


# *For 'TotalVisits', the 95% quantile is 10 whereas the maximum value is 251. Hence, we should cap these outliers at 95% value.*
#   *There are no significant outliers in 'Total Time Spent on Website'*
#   *For 'Page Views Per Visit', similar to 'TotalVisits', we should cap outliers at 95% value.*
# 
# We don't need to cap at 5% as the minimum value at 5% value are same for all the variables.

# In[120]:


# Outlier treatment
percentile = df_leads['TotalVisits'].quantile([0.95]).values
df_leads['TotalVisits'][df_leads['TotalVisits'] >= percentile[0]] = percentile[0]

percentile = df_leads['Page Views Per Visit'].quantile([0.95]).values
df_leads['Page Views Per Visit'][df_leads['Page Views Per Visit'] >= percentile[0]] = percentile[0]


# In[122]:


#Plot Boxplots to verify 
plt.figure(figsize=(15, 10))
for var in num_var:
   plt.subplot(3,1,num_var.index(var)+1)
   sns.boxplot(df_leads[var])
plt.show()


# In[123]:


# To plot numerical variables against target variable to analyse relations
plt.figure(figsize=(15, 5))
for var in num_var:
    plt.subplot(1,3,num_var.index(var)+1)
    sns.boxplot(y = var , x = 'Converted', data = df_leads)
plt.show()


# 1. 'TotalVisits' has same median values for both outputs of leads. No conclusion can be drwan from this.
# 2. People spending more time on the website are more likely to be converted. This is also aligned with our general knowledge.
# 3. 'Page Views Per Visit' also has same median values for both outputs of leads. Hence, inconclusive.

# **Visualising Categorical Variables**
# 

# In[124]:


# Categorical variables
cat_var = list(df_leads.columns[df_leads.dtypes == 'object'])
cat_var


# We saw percentage plots for categorical variables while cleaning the data. Here, we'll see these plots with respect to target variable 'Converted'
# 
# 

# In[125]:


# Functions to plot countplots for categorical variables with target variable

# For single plot
def plot_cat_var(var):
    plt.figure(figsize=(20, 7))
    sns.countplot(x = var, hue = "Converted", data = df_leads)
    plt.xticks(rotation = 90)
    plt.show()

# For multiple plots    
def plot_cat_vars(lst):
    l = int(len(lst)/2)
    plt.figure(figsize=(20, l*7))
    for var in lst:
        plt.subplot(l,2,lst.index(var)+1)
        sns.countplot(x = var, hue = "Converted", data = df_leads)
        plt.xticks(rotation = 90)
    plt.show()


# In[126]:


plot_cat_var(cat_var[0])


# Observations for Lead Origin :
# 1.'API' and 'Landing Page Submission' generate the most leads but have less conversion rates of around 30%. Whereas, 'Lead Add 2. Form' generates less leads but conversion rate is great. We should try to increase conversion rate for 'API' and 'Landing 3 3. Page Submission', and increase leads generation using 'Lead Add Form'. 'Lead Import' does not seem very significant.

# Observations for Lead Source :
# 
# 1. Spelling error: We've to change 'google' to 'Google'
# 2. As it can be seen from the graph, number of leads generated by many of the sources are negligible. There are sufficient numbers till Facebook. We can convert all others in one single category of 'Others'.
# 3. 'Direct Traffic' and 'Google' generate maximum number of leads while maximum conversion rate is achieved through 'Reference' and 'Welingak Website'.

# In[128]:


# To correct spelling error
df_leads['Lead Source'] = df_leads['Lead Source'].replace(['google'], 'Google')


# In[129]:


categories = df_leads['Lead Source'].unique()
categories


# We can see that we require first eight categories.
# 
# 

# In[130]:


# To reduce categories
df_leads['Lead Source'] = df_leads['Lead Source'].replace(categories[8:], 'Others')


# In[131]:


# To plot new categories
plot_cat_var(cat_var[1])


# In[132]:


plot_cat_vars([cat_var[2],cat_var[3]])


# Observations for Do Not Email and Do Not Call :
# 
# *As one can expect, most of the responses are 'No' for both the variables which generated most of the leads.

# In[133]:


plot_cat_var(cat_var[4])


# Observations for Last Activity :
# 
# 1. Highest number of lead are generated where the last activity is 'Email Opened' while maximum conversion rate is for the activity of 'SMS Sent'. Its conversion rate is significantly high.
# 2. Categories after the 'SMS Sent' have almost negligible effect. We can aggregate them all in one single category.

# In[134]:


categories = df_leads['Last Activity'].unique()
categories


# We can see that we do not require last five categories.
# 
# 

# In[136]:


# To reduce categories
df_leads['Last Activity'] = df_leads['Last Activity'].replace(categories[-5:], 'Others')
# To plot new categories
plot_cat_var(cat_var[4])


# In[137]:


plot_cat_var(cat_var[5])


# Observations for Country :
# Most of the responses are for India. Others are not significant.

# In[138]:


plot_cat_var(cat_var[6])


# Observations for Specialization :
# Conversion rates are mostly similar across different specializations.

# In[139]:


plot_cat_vars([cat_var[7],cat_var[8]])


# Observations for What is your current occupation and What matters most to you in choosing a course :
# 
# The highest conversion rate is for 'Working Professional'. High number of leads are generated for 'Unemployed' but conversion rate is low.
# Variable 'What matters most to you in choosing a course' has only one category with significant count.

# In[140]:


plot_cat_vars(cat_var[9:17])


# Observations for Search, Magazine, Newspaper Article, X Education Forums, Newspaper, Digital Advertisement, Through Recommendations, and Receive More Updates About Our Courses:
# As all the above variables have most of the values as no, nothing significant can be inferred from these plots.

# In[141]:


plot_cat_vars([cat_var[17],cat_var[18]])


# Observations for Tags and Lead Quality:
# 
# In Tags, categories after 'Interested in full time MBA' have very few leads generated, so we can combine them into one single category.
# Most leads generated and the highest conversion rate are both attributed to the tag 'Will revert after reading the email'.
# In Lead quality, as expected, 'Might be' as the highest conversion rate while 'Worst' has the lowest.

# In[142]:


categories = df_leads['Tags'].unique()
categories


# We can combine that last eight categories.
# 
# 

# In[143]:


# To reduce categories
df_leads['Tags'] = df_leads['Tags'].replace(categories[-8:], 'Others')
# To plot new categories
plot_cat_var(cat_var[17])


# In[144]:


plot_cat_vars(cat_var[19:25])


# Observations for Update me on Supply Chain Content, Get updates on DM Content, City, I agree to pay the amount through cheque, A free copy of Mastering The Interview, and Last Notable Activity :
# 
# 1. Most of these variables are insignificant in analysis as many of them only have one significant category 'NO'.
# 2. In City, most of the leads are generated for 'Mumbai'.
# 3. In 'A free copy of Mastering The Interview', both categories have similar conversion rates.
# 4. In 'Last Notable Activity', we can combine categories after 'SMS Sent' similar to the variable 'Last Activity'. It has most generated leads for the category 'Modified' while most conversion rate for 'SMS Sent' activity.

# In[145]:


categories = df_leads['Last Notable Activity'].unique()
categories


# We can see that we do not require last six categories.
# 
# 

# In[146]:


# To reduce categories
df_leads['Last Notable Activity'] = df_leads['Last Notable Activity'].replace(categories[-6:], 'Others')
# To plot new categories
plot_cat_var(cat_var[24])


# Based on the data visualization, we can drop the variables which are not significant for analysis and will not any information to the model.
# 
# 

# In[147]:


df_leads = df_leads.drop(['Do Not Call','Country','What matters most to you in choosing a course','Search','Magazine','Newspaper Article',
                          'X Education Forums','Newspaper','Digital Advertisement','Through Recommendations',
                          'Receive More Updates About Our Courses','Update me on Supply Chain Content',
                          'Get updates on DM Content','I agree to pay the amount through cheque',
                          'A free copy of Mastering The Interview'],1)
# Final dataframe
df_leads.head()


# In[149]:


df_leads.shape



# In[150]:


df_leads.info()


# In[151]:


df_leads.describe()


# **Data Preparation**

# In[152]:


# To convert binary variable (Yes/No) to 0/1
df_leads['Do Not Email'] = df_leads['Do Not Email'].map({'Yes': 1, 'No': 0})


# **Dummy Variable creation**
#  * For categorical variables with multiple levels, we create dummy features (one-hot encoded).

# In[153]:


# Categorical variables
cat_var = list(df_leads.columns[df_leads.dtypes == 'object'])
cat_var


# In[154]:


# To create dummy variables and drop first ones
dummy = pd.get_dummies(df_leads[cat_var], drop_first=True)

# To add result to the original dataframe
df_leads = pd.concat([df_leads, dummy], axis=1)

# To drop the original variables
df_leads = df_leads.drop(cat_var,1)


# In[155]:


df_leads.head()


# **Train-Test Split**

# In[156]:


# Importing required package
from sklearn.model_selection import train_test_split
# To put feature variable to X
X = df_leads.drop(['Converted'],axis=1)

X.head()


# In[157]:


# To put response variable to y
y = df_leads['Converted']

y.head()


# In[158]:


# To split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# **Feature Scaling**
# 

# In[159]:


# Importing required package
from sklearn.preprocessing import StandardScaler


# In[160]:


scaler = StandardScaler()


# In[161]:


# Numerical variables
num_var


# In[162]:


#Applying scaler to all numerical columns
X_train[num_var] = scaler.fit_transform(X_train[num_var])

X_train.head()


# In[163]:


# To check the conversion rate
conversion = (sum(df_leads['Converted'])/len(df_leads['Converted'].index))*100
conversion


# We have 37.85% conversion rate.
# 
# 

# **Building the Model**
# * After the creation of dummy variables, we have a large number of features. It is better to use RFE first for feature elimination.
# 
# *Feature Selection using RFE

# In[191]:


logis=sm.GLM(y_train,(sm.add_constant(X_train)),familt=sm.families.Binomial())
logis.fit().summary()


# In[192]:


# Instantiating

logreg = LogisticRegression()


# In[194]:


# Running rfe with different variable count

# Running with 19 variables

rfem = RFE(logreg, 19)
rfem = rfem.fit(X_train, y_train)


# In[195]:


list(zip(X_train.column# Checking for the true and false for the varibales after rfe

rfem.support_s, rfe.support_, rfe.ranking_))


# In[188]:


# Features selected
col = X_train.columns[rfe.support_]
col


# In[196]:


# Selecting the 'True' columns in rfem.support_

col = X_train.columns[rfem.support_]

X_train_1 = sm.add_constant(X_train[col]) # Adding constant


# Assessing the Model with StatsModels
# 

# In[190]:


import statsmodels.api as sm

# Function for building the model
def build_model(X,y):
    X_sm = sm.add_constant(X)    # To add a constant
    logm = sm.GLM(y, X_sm, family = sm.families.Binomial()).fit()    # To fit the model
    print(logm.summary())    # Summary of the model  
    return X_sm, logm


# In[184]:


from sklearn import metrics

# Function to get confusion matrix and accuracy
def conf_mat(Converted,predicted):
    confusion = metrics.confusion_matrix(Converted, predicted )
    print("Confusion Matrix:")
    print(confusion)
    print("Training Accuracy: ", metrics.accuracy_score(Converted, predicted))
    return confusion


# In[185]:


# Function for calculating metric beyond accuracy
def other_metrics(confusion):
    TP = confusion[1,1]    # True positives 
    TN = confusion[0,0]    # True negatives
    FP = confusion[0,1]    # False positives
    FN = confusion[1,0]    # False negatives
    print("Sensitivity: ", TP / float(TP+FN))
    print("Specificity: ", TN / float(TN+FP))
    print("False postive rate - predicting the lead conversion when the lead does not convert: ", FP/ float(TN+FP))
    print("Positive predictive value: ", TP / float(TP+FP))
    print("Negative predictive value: ", TN / float(TN+FN))


# Model 1
# Running the first model by using the features selected by RFE

# In[186]:


X1, logm1 = build_model(X_train[col],y_train)


# In[197]:


# To get predicted values on train set
y_train_pred_final = get_pred(X4,logm4)
y_train_pred_final.head()


# In[ ]:




