import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
import time
from sklearn import svm
from sklearn.metrics import classification_report

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(funcName)s :: %(lineno)d \:: %(message)s', level = logging.INFO)

products = pd.read_csv(r"E:\Internship\Forsk\Clothing Shoes And Jewelry\balanced_reviews.csv")
for i in range(0,len(products)-1):
    if type(products.iloc[i]['reviewText']) != str:
        products.iloc[i]['reviewText'] = str(products.iloc[i]['reviewText'])
products.dropna(inplace =  True)
products = products[products['overall'] != 3]
products['Sentiment'] = np.where(products['overall'] > 3, 1, 0)
print(products)

corpus = []
for i in range(0,products.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ' , products.iloc[i,1])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in stopwords.words('english')]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]    
    review = " ".join(review)
    corpus.append(review)        
products['corpus'] = corpus

print(products)

features_train, features_test, labels_train, labels_test = train_test_split(products['corpus'], products['Sentiment'], random_state = 42 )
vect = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True).fit(features_train)
features_train_tfidf_vectorized = vect.transform(features_train)
test_vectors = vect.transform(features_test)


# Perform classification with SVM, kernel=linear
SVC_classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
SVC_classifier_linear.fit(features_train_tfidf_vectorized, labels_train)
t1 = time.time()
prediction_linear = SVC_classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(labels_test, prediction_linear, output_dict=True)
print('positive: ', report['1'])
print('negative: ', report['0'])



file  = open(r"E:\Internship\Forsk\Clothing Shoes And Jewelry\SVM_classifier_model.pkl","wb")
pickle.dump(SVC_classifier_linear, file)
pickle.dump(vect.vocabulary_, open(r'E:\Internship\Forsk\Clothing Shoes And Jewelry\tfidf_features.pkl', 'wb'))

products.to_csv(r"E:\Internship\Forsk\Clothing Shoes And Jewelry\final_balanced_reviews.csv", index = False)