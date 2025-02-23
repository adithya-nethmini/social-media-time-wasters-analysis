# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

# Load the dataset

df = pd.read_csv("Time-Wasters on Social Media.csv")

# Display first few rows

print("First 5 rows of the dataset:")
print(df.head())

# Check data info and missing values

print("\nData Info:")
df.info()
print("\nMissing values count:")
print(df.isnull().sum())

# Identify numeric and categorical columns

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

# Fill missing numeric values with the median
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# Fill missing categorical values with the most frequent value (mode)
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Remove outliers using the IQR method (applied to numeric columns)

Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Exploratory Data Analysis (EDA)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Distribution of categorical variables

categorical_columns = [
    "Gender", "Location", "Owns Property", "Profession", "Demographics",
    "Platform", "Video Category", "Watch Reason", "DeviceType", "OS",
    "CurrentActivity", "ConnectionType"
]

for col in categorical_columns:
    print(f"\nDistribution of {col}:")
    print(df[col].value_counts())
    print("-"*50)

# Plotting Age Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution of Users")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot: Total Time Spent on Social Media by Platform

plt.figure(figsize=(10, 5))
sns.boxplot(x="Platform", y="Total Time Spent", data=df)
plt.xticks(rotation=45)
plt.title("Total Time Spent on Social Media by Platform")
plt.xlabel("Platform")
plt.ylabel("Total Time Spent")
plt.show()

# Scatterplot: Productivity Loss vs. Addiction Level colored by Platform

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Addiction Level", y="ProductivityLoss", hue="Platform", data=df)
plt.title("Correlation Between Addiction Level and Productivity Loss")
plt.xlabel("Addiction Level")
plt.ylabel("Productivity Loss")
plt.show()

# Correlation heatmap for numerical features

correlation_matrix = df.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Display strongest correlations with Total Time Spent

print("\nStrongest correlations with Total Time Spent:")
print(correlation_matrix["Total Time Spent"].sort_values(ascending=False))

# Clustering Analysis (K-Means)

# Selecting features for clustering
cluster_features = df[['Age', 'Total Time Spent', 'ProductivityLoss', 'Addiction Level']]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Applying K-Means clustering (with 3 clusters)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluate clustering with Silhouette Score

silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print("\nSilhouette Score for K-Means clustering:", silhouette_avg)

# Analyze cluster characteristics

cluster_means = df.groupby("Cluster")[["Age", "Total Time Spent", "ProductivityLoss", "Addiction Level"]].mean()
print("\nCluster Means")
print(cluster_means)

# Assigning meaningful cluster labels

def assign_cluster_label(cluster):
    if cluster == 0:
        return "Moderate Users (Balanced)"
    elif cluster == 1:
        return "Heavy Users (High Risk)"
    elif cluster == 2:
        return "Occasional Users (Low Engagement)"
    else:
        return "Unknown"

df['Cluster_Label'] = df['Cluster'].apply(assign_cluster_label)

# Visualize clusters based on Age and Total Time Spent

plt.figure(figsize=(8, 6))
sns.scatterplot(x="Age", y="Total Time Spent", hue="Cluster_Label", palette="Set1", data=df)
plt.title("User Clusters Based on Time Wasted on Social Media")
plt.xlabel("Age")
plt.ylabel("Total Time Spent")
plt.legend(title="User Segments")
plt.show()

# ANOVA test for Total Time Spent differences between clusters

group0 = df[df['Cluster'] == 0]['Total Time Spent']
group1 = df[df['Cluster'] == 1]['Total Time Spent']
group2 = df[df['Cluster'] == 2]['Total Time Spent']
f_stat, p_val = stats.f_oneway(group0, group1, group2)
print("\nANOVA F-statistic for Total Time Spent:", f_stat)
print("ANOVA p-value:", p_val)