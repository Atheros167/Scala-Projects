
import org.apache.spark.sql.SparkSession
import java.time.LocalDate
// Importing Machine Learning Libraries
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql
// Reduce the number of ERROR lines in the log files
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


val spark = SparkSession.builder().getOrCreate()

println("")

println("Loading all the csv files")
println("  1. Offer Table --> Offers")


val offers=(spark.read.option("header","true").option("inferSchema","true").format("csv")
  .load("D:\\Learning\\Research\\Projects\\valued_shoppers_kaggle\\data\\offers.csv"))

println("")
println("  2. Train Hist Table --> train_hist")
val train_hist = (spark.read.option("header","true").option("inferSchema","true").format("csv")
  .load("D:\\Learning\\Research\\Projects\\valued_shoppers_kaggle\\data\\trainHistory.csv"))


println("")
println("  3. Transaction Table --> txn_table")
val txn_table=(spark.read.option("header","true").option("inferSchema","true").format("csv")
  .load("D:\\Learning\\Research\\Projects\\valued_shoppers_kaggle\\data\\transactions.csv"))


println("")
println("Data Mining to get desired datadrame for merge with the train hist data")

// Here we can look at aggregating the sales and qty and other such attributes for the last three months
// But in this case we are looking at the complete year

println("")
println("******1. Create last Annual total sales per id(customer in this case) from the transaction table******")

val out_sum_sales=txn_table.groupBy("id").sum("purchaseamount").as("sumsales")
val out_sum_qty=txn_table.groupBy("id").sum("purchasequantity").as("totalqty")
val out_cnt_txns=txn_table.groupBy("id").count()


// Appending all the above tables to create one master train dataframe

val append_txn=(train_hist.join(out_sum_sales,Seq("id"),joinType = "left")
  .join(out_sum_qty,Seq("id"),joinType = "left")
  .join(out_cnt_txns,Seq("id"),joinType="left")
  .join(offers,Seq("offer"),joinType="left"))

println("")
println( "Renaming the new created fields from tables")
println("")
val append_txn2=(append_txn.withColumnRenamed("sum(purchaseamount)","total_amt").withColumnRenamed("sum(purchasequantity)","total_qty")
  .withColumnRenamed("count","total_txns"))


println("")
println("Our Target Variable is Categorical and a string. Converting to 0 and 1" )
println("")

println(" Importing all the machine Learning Libraries")
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.OneHotEncoder

// Can Also be done in a single line
//import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}

println("")
println("Since our predictor variable is a categorical, we need to replace them with 1 or 0")
println("")

val replace_repeater = udf {(repeater: String) =>  if(repeater == "t") 1 else 0}
val append_txn3=append_txn2.withColumn("repeater", replace_repeater(append_txn2("repeater")))

println("")
println("Assemble everything into one column called Feature")
println("")

val assembler=(new VectorAssembler()
                .setInputCols(Array("offer","id","chain","market","repeattrips","total_amt","total_qty","total_txns","category","offervalue","brand"))
                .setOutputCol("features"))

val logregdataall = (append_txn3.select(append_txn3("repeater").as("label"),$"offer",$"id",$"chain",$"market",
                      $"repeattrips",$"total_amt",$"total_qty",$"total_txns",$"category",$"offervalue",$"brand"))

val final_log_reg_data=logregdataall.na.drop()

println("")
println("Split the data into training and test data")
println("")

val Array(training,test) = final_log_reg_data.randomSplit(Array(0.7, 0.3), seed = 12345)

println("")
println("Need to set up a pipeline to set the workflow and execute")

import org.apache.spark.ml.Pipeline

println("")
println("Creating the Logistic Regression")
println("")

val lr = new LogisticRegression()


val pipeline = new Pipeline().setStages(Array(assembler, lr))

println("")
println("Training the Logistic Regression on the train Hist data")
println("")

val model = pipeline.fit(training)

println("")
println("Get Results on Test Set")
println("")

val results = model.transform(test)

////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////

println("")
println("Metrics and Evaluation")


import org.apache.spark.mllib.evaluation.MulticlassMetrics


println("Currently Spark2.1 does not support Metrics and Evaluation. So we need to use the RDD")

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd


println("")
println("Instantiate metrics object")


val metrics = new MulticlassMetrics(predictionAndLabels)


println("")
println("Confusion matrix")
println("")

println(metrics.confusionMatrix)