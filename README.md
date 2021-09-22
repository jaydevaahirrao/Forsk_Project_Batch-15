# Forsk_Project_Batch-15
data science project practicing

E-Commerce Store Review Classification using Deep Learning Techniques with Cloud Deployment:
 Creating a Web Based Dashboard to Gain Real Time Analysis on How Customers Feel About
Client's Product Using Machine Learning, Deep Learning and NLP Techniques with Data Processing on Large Dataset.

Client Expectation :
Since the client is in the business of Online Sale of Jewelry items through its
website and also TV Channels in USA, UK and Australia. They receive multiple feedback
and reviews of their products from all the channels, ( Website, TV Channels etc).
  The clients do not have a huge collection/ database of the reviews of their
products, so there is a challenge of training the model for Sentiments Analysis.
  They want us to develop a dashboard, through which they can Analyze the reviews
of their products and do a sentiment Analysis of it.
  Since there was a lack of enough data for training the model, we took a transfer
learning approach to solve it. We took the Jewelry data from Amazon Reviews data until
2018. It was 14 GB of JSON data.

To solve the client problem we need to create multiple small sub projects:

1. Process large dataset and convert into a smaller dataset after cleaning and
preprocessing the data to convert in to a balanced_reviews.csv
2. Read the balanced_reviews dataset and use the Bag of Words Model to train
a model and save into a pickle file.
3. Using Pre trained models to achieve the same.
4. Create a Web Scraper to scrape the data from Flipkart/etsy to test the newly
created model by saving the reviews/feedback into a database
5. Create a Dashboard using the Dash framework to integrate the picked file
and predict the sentiments of the stored data from the database.
