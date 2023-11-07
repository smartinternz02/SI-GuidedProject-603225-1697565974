import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix


import nltk
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
#nltk.download('stopwords')
df2=pd.read_csv(r"C:\Users\Lenovo\Downloads\archive (14)\Airline_Reviews.csv")


df2['Year_Flown'] = df2['Date Flown'].fillna('None').apply(lambda x: x[-4:])
for col in ['Year_Flown']:
    df2[col] = df2[col].fillna('None')
year_counts = df2['Year_Flown'].value_counts().sort_index()
print(year_counts)
x = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,np.nan]
y = [55,500,1261,2346,4709,9975,11809,10403,9994,12259,7057,1378,1022,4115,4980,3642]


# Replace NaN values with "Not Given"
x = [str(int(year)) if not np.isnan(year) else "Not Given" for year in x]

# Create the bar chart
plt.figure(figsize=(8, 5))
plt.bar(x, y, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Number of Airline Reviews in each year')
plt.xticks(rotation=60, ha="right")

# Show the plot
plt.tight_layout()



stopwords = set(stopwords.words('english'))

def remove_quotes(text):
    return re.sub(r'"', '', text)
def remove_stopwords(words):
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

stopwords.discard('not')

df2['Review_Title'] = df2['Review_Title'].apply(lambda x: x.lower())
df2['Review_Title'] = df2['Review_Title'].apply(remove_quotes)
df2['Review_Title'] = df2['Review_Title'].apply(lambda x: word_tokenize(x))
df2['Review_Title'] = df2['Review_Title'].apply(remove_stopwords)
df2['Review'] = df2['Review_Title']+df2['Review']
df2.drop(columns='Review_Title', inplace=True)
print(df2.head())
df4 = df2[['Airline Name', 'Overall_Rating', 'Review','Seat Type', 'Route', 'Date Flown','Seat Comfort', 'Cabin Staff Service', 'Food & Beverages','Inflight Entertainment','Ground Service','Value For Money','Recommended']]
print(df4.info())
print(df4.describe().T)
print(df4['Recommended'].value_counts())
print(df4.isnull().sum())

counts = df4.groupby(['Seat Type', 'Recommended']).size().unstack(fill_value=0)

plt.figure(figsize=(5, 4))
width = 0.35
x = range(len(counts.index))
plt.bar(x, counts['yes'], width, label='Recommended: Yes', color='skyblue', edgecolor='black')
plt.bar(x, counts['no'], width, label='Recommended: No', bottom=counts['yes'], color='lightcoral', edgecolor='black')
plt.title('Number of People in Each Seat Type by Recommendation')
plt.xlabel('Seat Type')
plt.ylabel('Count')
plt.xticks(x, counts.index, rotation=45)
plt.legend()
plt.tight_layout()


positive_recommendations = df4[df4['Recommended'] == 'yes']
airline_counts = positive_recommendations['Airline Name'].value_counts()
top_50_airlines = airline_counts.head(50)

plt.figure(figsize=(10, 8))
top_50_airlines.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 50 Airlines Based on Positive Recommendation Count')
plt.xlabel('Airline')
plt.ylabel('Count of Positive Recommendations')
plt.xticks(rotation=90)
plt.tight_layout()








from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['compound']
pd.options.mode.chained_assignment = None

df4['sentiment_score'] = df4['Review'].apply(lambda x: get_sentiment_score(x))

for column in df4.columns:
    print(f"First 5 values of column '{column}':")
    print(df4[column].head())
    print("\n")


df4 = df4.drop('Review', axis=1)

bin_edges = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]

# Create a histogram with custom bins
plt.figure(figsize=(6, 5))
plt.hist(df4['sentiment_score'], bins=bin_edges, edgecolor='black')
plt.xlabel('Sentiment Score Range')
plt.ylabel('Number of Scores')
plt.title('Sentiment Score Histogram')
plt.xticks(bin_edges)
#plt.grid(axis='y', linestyle='--', alpha=0.7
correlation_matrix = df4.iloc[:, :8].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix for Reviews and Ratings')
#plt.show()
correlation_matrix = df4[["sentiment_score", "Recommended"]].corr()
plt.figure(figsize=(4, 3.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#plt.title("Correlation Matrix between Sentiment Score and Recommended")
#plt.show()


total_rows = len(df4)
for column in df4.columns:
    null_count = df4[column].isnull().sum()
    null_percentage = (null_count / total_rows) * 100
    print(f"Percentage of null values in column '{column}': {null_percentage:.2f}%")


df4=df4.drop('Airline Name',axis=1)
df4=df4.drop('Route',axis=1)
df4=df4.drop('Date Flown',axis=1)
df4=df4.drop('Inflight Entertainment',axis=1)




columns_to_one_hot_encode = ['Seat Type', 'Recommended']

# Perform one-hot encoding for the specified columns
df4 = pd.get_dummies(df4, columns=columns_to_one_hot_encode)
print(df4.columns)
df4=df4.drop('Recommended_no',axis=1)
# List of columns to fill with median values
columns_to_fill_with_median = ['Overall_Rating', 'Seat Type_Business Class','Seat Type_Economy Class','Seat Type_First Class','Seat Type_Premium Economy','Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 'Value For Money']


# Fill NaN values in specified columns with the median
median_value = df4['Seat Type_Business Class'].median()
df4['Seat Type_Business Class'].fillna(median_value, inplace=True)
median_value = df4['Seat Type_Economy Class'].median()
df4['Seat Type_Economy Class'].fillna(median_value, inplace=True)
median_value = df4['Seat Type_First Class'].median()
df4['Seat Type_First Class'].fillna(median_value, inplace=True)
median_value = df4['Seat Type_Premium Economy'].median()
df4['Seat Type_Premium Economy'].fillna(median_value, inplace=True)
median_value = df4['Seat Comfort'].median()
df4['Seat Comfort'].fillna(median_value, inplace=True)
median_value = df4['Cabin Staff Service'].median()
df4['Cabin Staff Service'].fillna(median_value, inplace=True)
median_value = df4['Food & Beverages'].median()
df4['Food & Beverages'].fillna(median_value, inplace=True)
median_value = df4['Ground Service'].median()
df4['Ground Service'].fillna(median_value, inplace=True)
median_value = df4['Value For Money'].median()
df4['Value For Money'].fillna(median_value, inplace=True)


df4['Overall_Rating'] = df4['Overall_Rating'].replace('n', float('nan'))
median_value = df4['Overall_Rating'].median()
df4['Overall_Rating'].fillna(median_value, inplace=True)




from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming your target label is 'Recommended_yes' and you've already one-hot encoded your data
# Split the data into training and testing sets
X = df4.drop('Recommended_yes', axis=1)  # Features
y = df4['Recommended_yes']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

from sklearn.model_selection import GridSearchCV

# Define hyperparameters and their possible values
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # You can adjust the range
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

# Create a KNN classifier
knn_classifier = KNeighborsClassifier()

# Create a GridSearchCV object with cross-validation
#grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to your training data
knn_classifier.fit(X_train, y_train)

# Get the best hyperparameters
#best_params = grid_search.best_params_

# Get the best estimator (KNN model with best hyperparameters)
#best_knn_classifier = grid_search.best_estimator_

# Now, you can use the best KNN model for prediction
#y_pred_knn_tuned = best_knn_classifier.predict(X_test)

# Evaluate the tuned model's performance


# Calculate the accuracy score


# Print the results


# Logistic Regression



# Evaluation metrics
def predict_recommendation(overall_rating,seat_comfort,cabin_staff_service,food_and_beverages,ground_service,value_for_money,review,seat_type):
    # Get user input for the 7 features

    seat_type_features = [0, 0, 0,0]
    if seat_type == 'Economy':
        seat_type_features[0] = 1
    elif seat_type == 'First Class':
        seat_type_features[1] = 1
    elif seat_type == 'Premium Economy':
        seat_type_features[2] = 1
    else:
        seat_type_features[3]=1


    # Convert the review to sentiment score
    sentiment_score = get_sentiment_score(review)

    # Create a feature vector
    user_input = [overall_rating, seat_comfort, cabin_staff_service, food_and_beverages, ground_service, value_for_money,
                 sentiment_score] + seat_type_features

    # Make the prediction
    recommendation = knn_classifier.predict([user_input])

    # Display the prediction result
    if recommendation[0] == 1:
        return "yes"
    else:
        return "no"

# Call the function to predict a recommendation based on user input
"""

overall_rating = float(input("Enter Overall Rating: "))
seat_comfort = float(input("Enter Seat Comfort: "))
cabin_staff_service = float(input("Enter Cabin Staff Service: "))
food_and_beverages = float(input("Enter Food and Beverages: "))
ground_service = float(input("Enter Ground Service: "))
value_for_money = float(input("Enter Value for Money: "))
review = input("Enter Review: ")
seat_type = input("Enter Seat Type (Economy, First Class, Buisness or Premium Economy): ")
#predict_recommendation(overall_rating,seat_comfort,cabin_staff_service,food_and_beverages,ground_service,value_for_money,review,seat_type)
"""












