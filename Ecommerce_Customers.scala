//////////////////////////////////////////
////////    LINEAR REGRESSION  ///////////
/////////////////////////////////////////

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Starting a simple Spark Session

import org.spark.apache.sql.SparkSession
val spark=SparkSession.builder().getOrCreate()

// Reading in the Ecommerce Customers csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Ecommerce Customers")

// Printing out the schema of the dataframe Data

data.inferSchema()


////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// Since the data needs to have Labels and Features as columns for ML to recognize,
// we need to modify and create these fields


// Importing VectorAssembler and Vectors

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Label refers to the predictibg column. So we can easily rename the Yearly Amount Spent Column as "label"
// Also we will only be working with the numerical columns to create features

val data_mod1 = data.select(data("Yearly Amount Spent").as("label"),$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")


// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model
// We can then use VectorAssembler to convert the input columns of data_mod1
// to a single output column of an array called "features"
// Set the input columns from which we are supposed to read the values.
// Call this new object assembler
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")




// We can now use the assembler to transform our DataFrame to the two columns: label and features
val output = assembler.transform(datamod1).select($"label",$"features")

// Create a Linear Regression Model object
val lr = new LinearRegression()

// Fit the model to the data and call this model lrModel
val lrModel = lr.fit(output)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics!
// Use the .summary method off your model to create an object
// called trainingSummary
val trainingSummary = lrModel.summary

// Show the residuals, the RMSE, the MSE, and the R^2 Values.
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
