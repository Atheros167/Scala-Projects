
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Dataset

val spark=SparkSession.builder().getOrCreate()

val df1=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Export_2016_2017.csv").withColumn("Label",lit("Export")).withColumn("Year",lit(2016)))

val df2=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Export_2015_2016.csv").withColumn("Label",lit("Export")).withColumn("Year",lit(2015)))

val df3=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Export_2014_2015.csv").withColumn("Label",lit("Export")).withColumn("Year",lit(2014)))


val df4=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Import_2016_2017.csv").withColumn("Label",lit("Import")).withColumn("Year",lit(2016)))

val df5=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Import_2015_2016.csv").withColumn("Label",lit("Import")).withColumn("Year",lit(2015)))

val df6=(spark.read.option("header","true").option("inferSchema","true").
  format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat").
  load("/resources/data/Made_in_india/PC_Import_2014_2015.csv").withColumn("Label",lit("Import")).withColumn("Year",lit(2014)))

val all_data=df1.union(df2).union(df3).union(df4).union(df5).union(df6)


all_data.registerTempTable("all_data_table")

This is used for creating the graphs and plots in Zepplin notebook

%sql
  select country_name,sum(Export_value) as total_exports,sum(Import_value) as total_imports
from
(select country_name,year,Label,sum(case when Label='Export' then value else 0 end) Export_value,
sum(case when Label='Import' then value else 0 end) Import_value
from
all_data_table
group by country_name,Year,Label) a
group by country_name
order by total_exports desc limit 3


%sql

select pc_description,sum(Export_value) as total_exports,sum(Import_value) as total_imports
from
(select pc_description,year,Label,sum(case when Label='Export' then value else 0 end) Export_value,
sum(case when Label='Import' then value else 0 end) Import_value
from
all_data_table
where country_name='U S A'
group by pc_description,Year,Label) a
group by pc_description
order by total_exports desc limit 3
