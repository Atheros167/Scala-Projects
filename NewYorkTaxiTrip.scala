// Importing all the necessary libraries


import org.apache.spark.sql.SparkSession
import org.joda.time
import org.joda.time.format.DateTimeFormat
import org.joda.time.format.ISODateTimeFormat

// Creating a Spark Session
val spark=SparkSession.builder().getOrCreate()

// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator


// Reading the test dataset - This is the dataset to be scored using the model
val test=spark.read.option("header","true").option("inferSchema","true").format("csv").load("test.csv")
// 1. Calculating the distance between the trip
// 2. Calculating Hour of Pickup
// 3. Calculating the Weekday of pickup (Since this is a categorical string variable, we will have to use One Hot Encoder to convert this to dummy variables)

val test_temp=(test.withColumn("distance",hypot((test("dropoff_longitude")-test("pickup_longitude")),(test("dropoff_latitude")-test("pickup_latitude"))))
              .withColumn("hourOfDay",hour(test("pickup_datetime"))).withColumn("dayOfWeek",date_format(test("pickup_datetime"),"EEEE")))


val test_data=(test_temp.select("passenger_count","distance","hourOfDay","dayOfWeek","pickup_longitude","pickup_latitude","dropoff_latitude","dropoff_longitude"))

// Reading the test dataset
// 1. Calculating the distance between the trip
// 2. Calculating Hour of Pickup
// 3. Calculating the Weekday of pickup (Since this is a categorical string variable, we will have to use One Hot Encoder to convert this to dummy variables)

val train=spark.read.option("header","true").option("inferSchema","true").format("csv").load("train.csv")
val train_temp=(train.withColumn("distance",hypot((train("dropoff_longitude")-train("pickup_longitude")),(train("dropoff_latitude")-train("pickup_latitude"))))
  .withColumn("hourOfDay",hour(train("pickup_datetime"))).withColumn("dayOfWeek",date_format(train("pickup_datetime"),"EEEE"))
  .withColumn("label",train("trip_duration")))


// Here the data contains outliers and non meaningful records
// There are records where the trip length are equal to 1 second and a few where the trip has gone on for more than 1 million seconds
// Out of Convenience, I am keeping only the records that fall between the 5th percentile and the 99th percentile for modeling.
val quantiles = train_temp.stat.approxQuantile("label", Array(0.05,0.99),0.0)
val Q1 = quantiles(0)
val Q3 = quantiles(1)

// Keeping only the records where trip duration is within the percentiles in essence removing outliers the easy way.
val rem_outliers = train_temp.filter(s"label > $Q1 and label < $Q3")

// Dropping the NA/missing values
val train_data=(rem_outliers.select("label","passenger_count","distance","hourOfDay","dayOfWeek","pickup_longitude","pickup_latitude","dropoff_latitude","dropoff_longitude").na.drop())


// Importing the necessary libraries to convert the calculated dayOfweek categorical variable into a dummy variable (integer)
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val dayIndexer = new StringIndexer().setInputCol("dayOfWeek").setOutputCol("dayIndex").fit(train_data)
val indexed = dayIndexer.transform(train_data)
val dayEncoder = new OneHotEncoder().setInputCol("dayIndex").setOutputCol("dayVec")
val encoded_train = dayEncoder.transform(indexed)

val dayIndexer = new StringIndexer().setInputCol("dayOfWeek").setOutputCol("dayIndex").fit(test_data)
val indexed = dayIndexer.transform(test_data)
val dayEncoder = new OneHotEncoder().setInputCol("dayIndex").setOutputCol("dayVec")
val encoded_test = dayEncoder.transform(indexed)

///////////////////////////////////////////////////////////////////////////
//// Assemble everything together to be ("label","features") format ///////
///////////////////////////////////////////////////////////////////////////

val assembler = (new VectorAssembler()
  .setInputCols(Array("passenger_count","distance","hourOfDay","dayIndex","pickup_longitude","pickup_latitude","dropoff_latitude","dropoff_longitude"))
  .setOutputCol("features"))
val train_eval = assembler.transform(encoded_train).select($"label",$"features")
val test_eval = assembler.transform(encoded_test).select($"features")

////////////////////////////
/// Split the Data ////////
//////////////////////////
val Array(training_data, testing_data) = train_eval.randomSplit(Array(0.7, 0.3), seed = 12345)


/////////////////////////////////
//// Set Up the Pipeline ///////
////////////////////////////////
import org.apache.spark.ml.Pipeline

val lr = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)


// Fit the model to training data.
val lrModel = lr.fit(training_data)

// Get Results on Test Set
val results = lrModel.transform(testing_data)


//////////////////////////////////////////
//////// MODEL PERFORMANCE METRICS   /////
//////////////////////////////////////////

println("Print the coefficients and intercept for linear regression")
println("")
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics!
// Explore this in the spark-shell for more methods to call
val trainingSummary = lrModel.summary

println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

trainingSummary.residuals.show()

println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

// Output for the model above
//RMSE: 418.35580142042653
//r2: 0.457421596546823


val predict = lrModel.transform(test_eval)
