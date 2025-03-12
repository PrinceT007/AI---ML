# AI---ML
#This repository is dedicated to learning and exploring concepts related to Artificial Intelligence and Machine Learning. It contains curated resources, hands-on projects, code implementations, tutorials, and #datasets to build a strong foundation in AI & ML.

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Download the dataset from Kaggle (replace with your desired dataset)
!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009

# Unzip the dataset
!unzip red-wine-quality-cortez-et-al-2009.zip

# Data Preprocessing
# 1. Handle Missing Values (if any)
print("Missing values in the dataset:\n", df.isnull().sum())
# Replace missing values with mean or median or remove rows/columns
# Example: df['fixed acidity'].fillna(df['fixed acidity'].mean(), inplace=True)

# 2. Encode Categorical Features (if any)
# Example: If 'quality' is categorical, you can use one-hot encoding:
# df = pd.get_dummies(df, columns=['quality'], prefix=['quality'])

# 3. Feature Scaling
# Example: Scale numerical features using MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 4. Data Splitting
X = df.drop('quality', axis=1)  # Features
y = df['quality']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nPreprocessed Data:\n", df.head())


# prompt: Using the above data, explain me all possible visualization charts or graphs that can be derived using the libraries matplotlib and seaborn

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame from the provided code

# 1. Histograms: Distribution of individual features
for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)  # KDE for density estimation
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 2. Box plots: Identify outliers and distribution quartiles for each feature
for col in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
    plt.show()

# 3. Scatter plots: Relationship between pairs of features
sns.pairplot(df, hue='quality', vars=numerical_features) #hue for color coding by quality
plt.show()

# 4. Heatmap: Correlation matrix of all features
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()


# 5. Bar plots:  Frequency of quality levels
plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=df)
plt.title('Frequency of Wine Quality Levels')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()


# 6. Violin plots:  Combine box plot and kernel density estimation
for col in numerical_features:
  plt.figure(figsize=(8,6))
  sns.violinplot(x='quality', y=col, data=df)
  plt.title(f'Violin Plot of {col} vs. Quality')
  plt.show()

# 7. Jointplot:  Bivariate analysis with marginal distributions
for col in numerical_features:
  plt.figure(figsize=(8,6))
  sns.jointplot(x=col, y='quality', data=df, kind='kde') #kde for density
  plt.suptitle(f'Joint Plot of {col} vs. Quality', y=1.02) #adjust title position
  plt.show()

# 8.  Lmplot:  Linear regression model fits
for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.lmplot(x=col, y='quality', data=df)
    plt.title(f'Linear Regression Plot of {col} vs. Quality')
    plt.show()

