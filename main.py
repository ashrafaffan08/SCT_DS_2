# ================================
# Task 02: Titanic Dataset - Data Cleaning & EDA
# ================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Plot Style
sns.set(style="whitegrid")

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# ================================
# Basic Information
# ================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ================================
# Data Cleaning
# ================================

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it contains many missing values
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ================================
# Exploratory Data Analysis (EDA)
# ================================

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df)
plt.title("Age vs Fare")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# Findings
# ================================

print("\n========== KEY FINDINGS ==========")

print("1. Female passengers had a much higher survival rate than male passengers.")
print("2. First-class passengers survived more frequently than second- and third-class passengers.")
print("3. Most passengers traveled in third class.")
print("4. Younger passengers showed a slightly higher survival rate.")
print("5. Passengers who paid higher fares were more likely to survive.")
print("6. The Cabin column contained many missing values, so it was removed.")
print("7. Missing values in Age and Embarked were filled using median and mode respectively.")

print("\n========== CONCLUSION ==========")
print("The Titanic dataset was cleaned by handling missing values and removing the Cabin column. Exploratory Data Analysis revealed that gender, passenger class, age, and fare were the most influential factors affecting passenger survival.")