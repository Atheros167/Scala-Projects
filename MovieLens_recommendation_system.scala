
// Import statements to start a spark session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().appName("Recommendation_system").getOrCreate()

// Loading the csv Dataset "ratings" and other associated files
val movie_rating=spark.read.option("header","true").option("InferSchema","true").format("csv").load("ratings.csv")
val movies_labels=spark.read.option("header","true").option("inferSchema","true").format("csv").load("movies.csv")
val tags=spark.read.option("header","true").option("inferSchema","true").format("csv").load("tags.csv")

// Check out the schema of the dataset

movie_rating.printSchema

// |-- userId: integer (nullable = true)
// |-- movieId: integer (nullable = true)
// |-- rating: double (nullable = true)
// |-- timestamp: integer (nullable = true)


// Splitting the data set into training and test

val Array(movie_train,movie_test)=movie_rating.randomSplit(Array(0.7,0.3))

// Using the ALS to create a model for recommendation om the train dataset
import org.apache.spark.ml.recommendation.ALS
val als=new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setRatingCol("rating").setItemCol("movieId")

val Model=als.fit(movie_train)

// Now that the model is build we need to implement that model on the test dataset to compute how it performs

val predict= Model.transform(movie_test)


// Here we can use several different measuring/evaluation metrics like RMSE,MSE etc
// I am using ABS value to evaluate

import org.apache.spark.sql.functions._
val error=predict.select(abs($"rating"-$"prediction"))

// Now we can check the mean absolute error using describe function

error.na.drop().describe().show()


//  +-------+--------------------------+
//  |summary|abs((rating - prediction))|
//  +-------+--------------------------+
//  |  count|                   5996726|
//  |   mean|        0.6298020868818786|
//  | stddev|        0.5256640954706371|
//  |    min|                       0.0|
//  |    max|         11.77287769317627|
//  +-------+--------------------------+


// The above table shows that the average absolute difference between the actual rating
// by a user for a movie versus the predicted rating is 0.63



// From this we can Generate top 10 movie recommendations for each user
val userRecs = model.recommendForAllUsers(10)
userRecs.show(9)
//+------+--------------------+
//|userId|     recommendations|
//+------+--------------------+
//|   148|[[82051,13.087453...|
//|   463|[[96255,12.827446...|
//|   471|[[96255,13.943589...|
//|   496|[[73529,12.144476...|
//|   833|[[116951,14.33647...|
//|  1088|[[96941,13.815263...|
//|  1238|[[89632,10.140041...|
//|  1342|[[112907,10.96657...|
//|  1580|[[116951,16.65443...|


// Generate top 10 user recommendations for each movie
val movieRecs = model.recommendForAllItems(10)


