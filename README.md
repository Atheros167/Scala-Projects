# Scala-Projects

## List of Projects on Scala

#### **Data Visualization Project**
1. This project was to showcase some of the cool visualizations that are possible with Scala
2. This imports data obtained from [DVP](https://www.kaggle.com/hiteshp/make-in-india/data) 
   to do some dashboarding and visualization exercises.The dataset details the Imports and Exports from India for three years (2014-2016) to various countries.From this we can build a few simple plots and summarizations
3. There are a lot of possible graphs, I have tried Pie Chart and Bar Graph
4. The URL to work interactively on this is available here
[Results](https://my.datascientistworkbench.com/tools/zeppelin-notebook)

    

#### **String_Compare**

1. Check if any alphabet in one word exists in another word
2. Scope of the project
    Identify how many of the words are common in the two words
    Which characters are common?
    Index of the common characters.
    Toggle the cases of the common characters in both words
 
 Here is the link to the Zepl Online Notebook : 
 [Str_comp](https://www.zepl.com/spaces/S_ZEPL/1d718b29bf21413fbaefc13318dd51af)

#### **Fibonacci Sequence**
1. Learning to work with integers and
2. Creating Objects with Main and
3. Creating Objects without Main
4. Here is the link to Zepl Online Notebook : 
[Fib_Seq](https://www.zepl.com/spaces/S_ZEPL/2f367acdce714ae990a9baedf0829479)
   

#### **Kaggle Value Shopper - Logistic Regression**

1. This is a Kaggle Competition located at: 
[Val_shop](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)
    
    The dataset includes the Offer Table with details related to the Offers for coupons. The large transaction table holds 1 years worth of transaction history. There are two separate datasets - Training Dataset and Test Dataset.

    The purpose of this project is to use the available data to build a Logistic Regression model that uses the training data to create a model that can be used to score the test dataset. The output is to predict the likelihood of a customer using the coupon.


    
#### **Ecommerce Customer - Linear Regression**

1. We have a dataset containing customer information for a ecommerce company
2. Our Objective here is to use the available data:
    Transform the data into Labels and Features for ML library to use
    Apply Linear Regression on the data to predict the yearly spend by a customer.
    Print out the metrics of the Regression models
3. The data set is included as Ecommerce_customer.csv
4. The dataset is available in the [Data](https://github.com/Atheros167/Scala-Projects/blob/master/Ecommerce%20Customers.csv)
5. The code is available at the [Code_lin_reg](https://github.com/Atheros167/Scala-Projects/blob/master/Ecommerce_Customers.scala)


#### Unsupervised Clustering - K Means Clustering

1. The objective is to use the data available to correctly predict the region from the following 7 input variables:
   1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
   2)	MILK: annual spending (m.u.) on milk products (Continuous);
   3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
   4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
   5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
   6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
   7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
2. The data is available at [Clus_data](http://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
3. The code is available at [Clus_code](https://github.com/Atheros167/Scala-Projects/blob/master/Kmeans_wholesale_customers.scala)



#### Principal Component Analysis

1. The objective is to use the dataset obtained from Breast Cancer Wisconsin (Diagnostic) Database to perform PCA on the data.
2. Here is a description of the dataset used:

   Number of Instances: 569
   Number of Attributes: 30 numeric, predictive attributes and the class
   :Attribute Information:
       - radius (mean of distances from center to points on the perimeter)
       - texture (standard deviation of gray-scale values)
       - perimeter
       - area
       - smoothness (local variation in radius lengths)
       - compactness (perimeter^2 / area - 1.0)
       - concavity (severity of concave portions of the contour)
       - concave points (number of concave portions of the contour)
       - symmetry
       - fractal dimension ("coastline approximation" - 1)
   
The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.  For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

3. The code is available at [PCA_code](https://github.com/Atheros167/Scala-Projects/blob/master/PCA_Cancer_data.scala)
4. The data is available at [PCA_data](https://github.com/Atheros167/Scala-Projects/blob/master/Cancer_Data.csv)
         
#### Recommendation Engine

1. The Objective of this code is to build out a recommendation engine from the MovieLens dataset (ML20M) located at [MovieLens_data](https://grouplens.org/datasets/movielens/) to read the User_id, Movie_id and Rating Information with 20Million Movie ratings and create a ALS recommendation Algorithm
2. Use the ALS model to build and score the test set (30% of the input data) to measure/evaluate the model
3. The output code is located at: [MLens_Code](https://github.com/Atheros167/Scala-Projects/blob/master/MovieLens_recommendation_system.scala)


#### Kaggle Competition - New York Taxi Trip

1. The competition is live as of now and the objective is to predict the trip duration for the test dataset provided using a model built out of train dataset.
2. The competition can be found at [NYTTrip](https://www.kaggle.com/c/nyc-taxi-trip-duration)
3. Built a simple Linear Regression to predict the trip duration. [NYTT_Code](https://github.com/Atheros167/Scala-Projects/blob/master/NewYorkTaxiTrip.scala)
* Resulting RSquare of 0.45 can be improved.
* Next steps to improve the model would be to explore other model fits like Neural Networks
