/////////////////////////////////
// K MEANS PROJECT EXERCISE ////
///////////////////////////////

// Objective: to cluster clients of a Wholesale Distributor
// based off of the sales of some product categories

// Source of the Data
//http://archive.ics.uci.edu/ml/datasets/Wholesale+customers


///////////////////////////////////////////
//// CLUSTERING _ K MEANS CLUSTERING //////
///////////////////////////////////////////

// Import SparkSession
import org.apache.spark.sql.SparkSession

// Setting the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session Instance
val spark = SparkSession.builder().getOrCreate()

// Import Kmeans clustering Algorithm
import org.apache.spark.ml.clustering.KMeans

// Load the Wholesale Customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

// Select the following columns for the training set:
// Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
// Cal this new subset feature_data
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")


// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features

val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

// Using the assembler object to transform the feature_data

val training_data = assembler.transform(feature_data).select("features")

// Create a Kmeans Model with K=3
val kmeans = new KMeans().setK(3).setSeed(1L)

// Fit that model to the training_data
val model = kmeans.fit(training_data)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(training_data)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
