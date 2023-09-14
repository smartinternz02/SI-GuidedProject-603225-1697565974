import pandas as pd
df=pd.read_csv(r"C:\Users\Lenovo\Downloads\penguins_size.csv")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Univariate Analysis

import matplotlib.pyplot as plt
import seaborn as sns
numeric_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.show()
categorical_features = ['species', 'island', 'sex']
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=feature)
    plt.title(f'Bar plot of {feature}')
    plt.xticks(rotation=45)
    plt.show()

#Bivariate Analysis

sns.pairplot(df, hue='species')
plt.show()
for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='species', y=feature)
    plt.title(f'Box plot of {feature} by Species')
    plt.xticks(rotation=45)
    plt.show()

#Multivariate Analysis

sns.pairplot(df, hue='species')
plt.show()

#Desciptive statistics

full_stats = df.describe(include='all')

#Null values

missing_values = df.isnull().sum()
print(missing_values)
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
for column in numeric_columns:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
mode_value = df['sex'].mode().iloc[0]
df['sex'].fillna(mode_value, inplace=True)
missing_values = df.isnull().sum()
print(missing_values)

#Check Correlation with the Target (Species)

correlation_matrix = df.corr()
target_correlation = correlation_matrix['species'].sort_values(ascending=False)
print(target_correlation)

#one hot encoding

df_encoded = pd.get_dummies(df, columns=['species', 'island', 'sex'])
print(df_encoded.head())

#splitting the data

X = df_encoded.drop(columns=['species_Chinstrap', 'species_Adélie', 'species_Gentoo'])
y = df_encoded[['species_Chinstrap', 'species_Adélie', 'species_Gentoo']]

# Scaling the data using StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



