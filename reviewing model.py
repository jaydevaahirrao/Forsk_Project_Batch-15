#importing the necessary libraries 
import pandas as pd
import numpy as np
from glob import glob
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle


df_reader = pd.read_json(r":\Forsk Files\Gift_Cards.json", lines = True, chunksize = 10000 )

#creating the balanced csv
counter = 1
for chunk in df_reader:
    new_df = pd.DataFrame(chunk[['overall', 'reviewText','summary']])
    new_df1 = new_df[new_df['overall'] == 1].sample(40)
    new_df2 = new_df[new_df['overall'] == 2].sample(40)
    new_df3 = new_df[new_df['overall'] == 4].sample(40)
    new_df4 = new_df[new_df['overall'] == 5].sample(40)
    new_df5 = new_df[new_df['overall'] == 3].sample(80)
    new_df6 = pd.concat([new_df1, new_df2, new_df3, new_df4, new_df5], axis = 0,ignore_index = True)
    new_df6.to_csv("D:/Forsk Files/" + str(counter) + ".csv", index = False)
    counter = counter+1
        
#getting  all the csv files
#['1.csv','2.csv',..........,'33.csv']
filenames = glob('D:/Forsk Files/*.csv')
dataframes = []

for f in filenames:
    dataframes.append(pd.read_csv(f))

finaldf = pd.concat(dataframes, axis = 0, ignore_index = True)
finaldf.to_csv("D:/Forsk Files/balanced_reviews_gift.csv", index = False)

#reading the csv file
df = pd.read_csv('D:/Forsk Files/balanced_reviews_gift.csv')

#handle the missing data
df.dropna(inplace =  True)

#leaving the reviews with rating 3 and collect reviews with
df = df [df['overall'] != 3]

#creating a label
df['Positivity'] = np.where(df['overall'] > 3 , 1 , 0)

#cleaning data
corpus = []
for i in range(0,df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ' , df.iloc[i,1])
    review = review.lower()
    review = review.split()
    #remove the stopwords
    review = [word for word in review if not word in stopwords.words('english')]
    #stemming
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]    
    review = " ".join(review)
    corpus.append(review)

features_vectorized = CountVectorizer().fit_transform(corpus)
labels = df.iloc[:,-1]

#train tset split
features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )

#tfidf vectarization
vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)

#logistic regression
model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))

#checking the model's score
roc_auc_score(labels_test, predictions)

#pickling the model
file  = open("pickle_model.pkl","wb")
pickle.dump(model, file)

#pickling the vocabulary
pickle.dump(vect.vocabulary_, open('features.pkl', 'wb'))