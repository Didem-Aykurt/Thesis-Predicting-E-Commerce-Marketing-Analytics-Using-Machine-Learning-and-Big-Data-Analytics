#!/usr/bin/env python
# coding: utf-8

# ## Thesis: Predicting E-Commerce Marketing Analytics Using Machine Learning and Big Data Analytics Code Paper
# ### Didem B. Aykurt
# ### Colorado State University Global
# ### MIS581: Business Intelligence and Data Analytics
# ### Dr. Steve Chung
# ### January 14,2024
# 

# In[ ]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[2]:


# Load the datasets
online_sales = pd.read_csv('Online_Sales.csv')
customers_data = pd.read_excel('CustomersData.xlsx', sheet_name='Customers')
discount_coupon = pd.read_csv('Discount_Coupon.csv')
marketing_spend = pd.read_csv('Marketing_Spend.csv')
tax_amount = pd.read_excel('Tax_amount.xlsx', sheet_name='GSTDetails')


# In[3]:


# Display basic information about the datasets
print(online_sales.info())
print(customers_data.info())
print(discount_coupon.info())
print(marketing_spend.info())
print(tax_amount.info())


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
# Assuming 'Avg_Price' is the numeric variable in the 'online_sales' dataset
numeric_variable = 'Avg_Price'
plt.figure(figsize=(50, 5))
sns.set(style="whitegrid")


# In[5]:


# Plot the histogram with count annotations",
ax = sns.histplot(online_sales[numeric_variable], kde=True, bins=30)
for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
plt.title(f'Distribution of {numeric_variable}')
plt.xlabel(numeric_variable)
plt.ylabel('Frequency')
plt.show()


# In[6]:


# Visualize categorical variables in Online Sales dataset
plt.figure(figsize=(12, 6))
sns.countplot(x='Product_Category', data=online_sales, hue='Coupon_Status')
plt.title('Count of Products by Category and Coupon Status - Online Sales Dataset')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[7]:


# Assuming 'Coupon_Status' is a categorical variable in the 'Online_Sales' dataset\n",
online_sales = pd.read_csv('Online_Sales.csv')
# Count the occurrences of each coupon status
coupon_counts = online_sales['Coupon_Status'].value_counts()
    
print(coupon_counts)
#Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(coupon_counts, labels=coupon_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
plt.title('Distribution of Coupon Status')
plt.show()


# In[8]:


# Assuming 'Tenure_Months' is the variable in the 'customers_data' dataset
plt.figure(figsize=(8, 5))
sns.histplot(customers_data['Tenure_Months'], kde=True)
plt.title('Distribution of Tenure in Months - Customers Data Dataset')
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.show()
# Print summary statistics of the 'Tenure_Months' variable
tenure_stats = customers_data['Tenure_Months'].describe()
print('Summary Statistics of Tenure_Months')
print(tenure_stats)


# In[9]:


# Assuming 'Discount_pct' is the variable in the 'discount_coupon' dataset
plt.figure(figsize=(8, 5))
sns.histplot(discount_coupon['Discount_pct'], kde=True)
plt.title('Distribution of Discount Percentage - Discount Coupon Dataset')
plt.xlabel('Discount Percentage')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Group by discount percentage and count the number of coupons for each percentage
coupon_counts_by_percentage = discount_coupon['Discount_pct'].value_counts().reset_index()
coupon_counts_by_percentage.columns = ['Discount Percentage', 'Coupon Count']
print('Coupon Counts by Discount Percentage')
print(coupon_counts_by_percentage)


# In[11]:


# Assuming 'Product_Description' is the variable representing product names in the 'Online_Sales' dataset
online_sales = pd.read_csv('Online_Sales.csv')
# Calculate the top products by counting occurrences
top_products = online_sales['Product_Description'].value_counts().nlargest(10)
# Create a bar chart for the top products
plt.figure(figsize=(12, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products by Sales')
plt.xlabel('Product Description')
plt.ylabel('Number of Items Sold')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()
print(top_products)


# In[26]:


online_sales['Transaction_Date'] = pd.to_datetime(online_sales['Transaction_Date'])

# Create a new column for 'Revenue' by multiplying 'Quantity' and 'Avg_Price'
online_sales['Revenue'] = online_sales['Quantity'] * online_sales['Avg_Price']

# Group by month and sum the revenue for each month
monthly_revenue = online_sales.groupby(online_sales['Transaction_Date'].dt.to_period("M"))['Revenue'].sum().reset_index()

# Convert the 'Transaction_Date' back to datetime for plotting
monthly_revenue['Transaction_Date'] = monthly_revenue['Transaction_Date'].dt.to_timestamp()

# Print the monthly revenue
print("Monthly Revenue:")
print(monthly_revenue)

# Visualize the monthly revenue
plt.figure(figsize=(12, 6))
plt.bar(monthly_revenue['Transaction_Date'], monthly_revenue['Revenue'], color='skyblue', width=20)
plt.title('Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[13]:


print(online_sales['Transaction_Date'])


# In[14]:


# Make sure the 'Date' column is in datetime format
marketing_spend['Date'] = pd.to_datetime(marketing_spend['Date'])
# Create a new column for the month
marketing_spend['Month'] = marketing_spend.Date.dt.month
# Select only the numeric columns
numeric_cols = marketing_spend.select_dtypes(['int', 'float']).columns
# Group by month and sum the values
monthly_spend = marketing_spend.groupby(marketing_spend.Date.dt.month)[numeric_cols].sum()
# Plot the data
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_spend.index.astype(str), y=monthly_spend['Offline_Spend'], label='Offline Spend')
sns.lineplot(x=monthly_spend.index.astype(str), y=monthly_spend['Online_Spend'], label='Online Spend')
plt.title('Monthly Marketing Spend - Marketing Spend Dataset')
plt.xlabel('Month')
plt.ylabel('Marketing Spend')
plt.legend()
plt.show()


# In[15]:


# Visualize GST Percentage distribution in Tax Amount dataset
plt.figure(figsize=(8, 5))
sns.barplot(x='Product_Category', y='GST', data=tax_amount)
plt.title('Average GST Percentage by Product Category - Tax Amount Dataset')
plt.xticks(rotation=90)
plt.xlabel('Product Category')
plt.ylabel('Average GST Percentage')
plt.show()


# In[16]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Select relevant features for clustering
features_for_clustering = customers_data[['Tenure_Months']]
# Standardize the features
scaler = StandardScaler()
features_for_clustering_scaled = scaler.fit_transform(features_for_clustering)
# Determine optimal number of clusters using silhouette score
best_num_clusters = 2  # Set the initial number of clusters
best_silhouette_score = -1
#initialize kmeans parameters
k_rng=range(2,6)
for num_clusters in k_rng:  # Adjust the range as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_for_clustering_scaled)
    silhouette_avg = silhouette_score(features_for_clustering_scaled, cluster_labels)
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_num_clusters = num_clusters
# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_num_clusters, random_state=42)
customers_data['Cluster_Label'] = kmeans.fit_predict(features_for_clustering_scaled)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
# Print the count of customers in each cluster
cluster_counts = customers_data['Cluster_Label'].value_counts()
print("Count of Customers in Each Cluster:")
print(cluster_counts)
# Plotting the distribution of customers across clusters
plt.figure(figsize=(5, 3))
sns.countplot(x='Cluster_Label', data=customers_data, palette='viridis')
plt.title('Customer Distribution Across Clusters')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Customers')
plt.show()


# In[18]:


print(customers_data.Cluster_Label)


# In[19]:


import pandas as pd
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm

# Merge datasets as needed
# For example, if comparing purchasing behavior between customer segments
merged_data = pd.merge(online_sales, customers_data, on='CustomerID', how='inner')
# Extract relevant columns for hypothesis testing
grouped_data = merged_data.groupby('Cluster_Label')['Quantity'].apply(list)
# Perform ANOVA for Customer Segmentation
anova_result = f_oneway(*grouped_data)
# Print ANOVA result
print("ANOVA Result for Customer Segmentation:")
print(anova_result)


# In[20]:


marketing_spend['Date'] = pd.to_datetime(marketing_spend['Date'])
online_sales['Transaction_Date'] = pd.to_datetime(online_sales['Transaction_Date'])
# Merge datasets on the 'Date' column
merged_data_monthly = pd.merge(online_sales, marketing_spend, left_on='Transaction_Date', right_on='Date', how='inner')
# Group by month and calculate correlation for each month
correlation_per_month = []
months = merged_data_monthly['Date'].dt.month.unique()
for month in months:
        monthly_data = merged_data_monthly[merged_data_monthly['Date'].dt.month == month]
        correlation, p_value = pearsonr(monthly_data['Online_Spend'], monthly_data['Quantity'])
        correlation_per_month.append((month, correlation, p_value))
    
# Create a DataFrame from the correlation results
correlation_df = pd.DataFrame(correlation_per_month, columns=['Month', 'Correlation', 'P-value'])
    
# Print the correlation results
print("Correlation Results for Each Month:")
print(correlation_df)
    
# Visualize the correlation results
plt.figure(figsize=(6, 4))
plt.plot(correlation_df['Month'], correlation_df['Correlation'], marker='o', linestyle='-', color='b')
plt.title('Correlation between Online Spend and Quantity - Monthly')
plt.xlabel('Month')
plt.ylabel('Correlation Coefficient')
plt.xticks(months)  # Ensure all months are displayed on the x-axis
plt.show()


# In[21]:


# Extract relevant columns for ANOVA
grouped_data_monthly = [merged_data_monthly[merged_data_monthly['Date'].dt.month == month]['Quantity'] for month in months]
# Perform ANOVA
anova_result_monthly = f_oneway(*grouped_data_monthly)
# Print ANOVA result
print("ANOVA Result for Marketing Effectiveness Hypotheses (H2):")
print(anova_result_monthly)


# In[22]:


#Anova test for GST by product category
# For example, if comparing GST between product category
merged_data_tax = pd.merge(merged_data, tax_amount, on='Product_Category', how='inner')
# Extract relevant columns for hypothesis testing
grouped_data = merged_data_tax.groupby('Cluster_Label')['GST'].apply(list)
# Perform ANOVA for Customer Segmentation
anova_result = f_oneway(*grouped_data)
# Print ANOVA result
print("ANOVA Result for  GST Impact on Prodcut Category Hypotheses (H3):")
print(anova_result)


# In[24]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

merged_data_tax_next = pd.merge(online_sales, tax_amount, on='Product_Category', how='inner')
#clean null variable
df_clean = merged_data_tax_next.dropna(axis=0, how="any")
# Create a two-way ANOVA model
model =smf.ols('Quantity ~ C(GST) + C(Product_Category) + C(GST):C(Product_Category)', data=df_clean).fit()

# Generate an ANOVA table
anova_table = sm.stats.anova_lm(model, typ=3)

# Print the ANOVA table
print("ANOVA Result for GST Impact on Product Categories Hypotheses (H3):")
print(anova_table)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Assuming 'Date' is present in the 'online_sales' dataset
online_sales['Transaction_Date'] = pd.to_datetime(online_sales['Transaction_Date'])
# Create a new column for the month of the first purchase (cohort)
online_sales['Cohort_Month'] = online_sales.groupby('CustomerID')['Transaction_Date'].transform('min').dt.to_period("M")
# Label customers as retained (1) or not retained (0)
online_sales['Retained'] = online_sales.groupby('CustomerID')['Transaction_Date'].transform(lambda x: x.diff().max() > pd.Timedelta('30 days')).astype(int)
# Feature engineering
features = pd.get_dummies(online_sales['Product_Category'], prefix='Product_Category')
features['Total_Spend'] = online_sales.groupby('CustomerID')['Avg_Price'].transform('sum')
features['Total_Quantity'] = online_sales.groupby('CustomerID')['Quantity'].transform('sum')
# Select features and target variable
X = features
y = online_sales['Retained']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build a predictive model (Logistic Regression)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# In[ ]:




