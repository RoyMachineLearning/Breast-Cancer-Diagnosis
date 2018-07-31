// Databricks notebook source
// MAGIC %md
// MAGIC <pre>
// MAGIC <b><u>Project Details : Breast Cancer Diagnosis in Wisconsin</b></u>
// MAGIC 
// MAGIC Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
// MAGIC 
// MAGIC The 3-dimensional space is that described in: 
// MAGIC [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
// MAGIC 
// MAGIC This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/
// MAGIC 
// MAGIC Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
// MAGIC 
// MAGIC Attribute Information:
// MAGIC 
// MAGIC 1) ID number 2) Diagnosis (M = malignant, B = benign)
// MAGIC 
// MAGIC Ten real-valued features are computed for each cell nucleus:
// MAGIC 
// MAGIC a) radius (mean of distances from center to points on the perimeter) 
// MAGIC b) texture (standard deviation of gray-scale values) 
// MAGIC c) perimeter 
// MAGIC d) area 
// MAGIC e) smoothness (local variation in radius lengths) 
// MAGIC f) compactness (perimeter^2 / area - 1.0) 
// MAGIC g) concavity (severity of concave portions of the contour) 
// MAGIC h) concave points (number of concave portions of the contour) 
// MAGIC i) symmetry 
// MAGIC j) fractal dimension ("coastline approximation" - 1)
// MAGIC 
// MAGIC The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
// MAGIC 
// MAGIC All feature values are recoded with four significant digits.
// MAGIC 
// MAGIC Missing attribute values: none
// MAGIC Class distribution: 357 benign, 212 malignant</pre>

// COMMAND ----------

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SQLContext, Row, DataFrame, Column}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.DenseVector

import org.apache.spark.ml.feature.{Imputer, ImputerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, IndexToString} 
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}

import org.apache.spark.ml.feature.{Bucketizer,Normalizer}

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel,DecisionTreeClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.NaiveBayes

//Implement PCA
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable
import com.microsoft.ml.spark.{LightGBMClassifier,LightGBMClassificationModel}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoostClassificationModel}

// COMMAND ----------

// Setup the Scala syntax for Spark SQL
val sparkImplicits = spark
import sparkImplicits.implicits._

// COMMAND ----------

val DF = (sqlContext.read
                  .option("header","true")
                  .option("inferSchema","true")
                  .format("csv")
                  .load("/FileStore/tables/Cancer_Data/Cancer_Data.csv"))


// COMMAND ----------

DF.printSchema()

// COMMAND ----------

display(DF)

// COMMAND ----------

// MAGIC %md Analyzing the data : Check for Missing Values

// COMMAND ----------

val check_90_null = udf { xs: Seq[String] =>
  xs.count(_ == null) >= (xs.length * 0.9)
}

val columns = array(DF.columns.map(col): _*)

val summary = DF.filter(not(check_90_null(columns))).describe()

//display summary statistics - NO MISSING VALUES FOUND
display(summary)

// COMMAND ----------

// MAGIC %md Convert Diagnostics to Binary : benign = 0 or malignant = 1 

// COMMAND ----------

val DF_features =  DF.withColumn("Int_Diagnostics", when($"diagnosis" === "M",1).otherwise(0))
display(DF_features)

// COMMAND ----------

// MAGIC %md Exploratory Data Analysis for the entire dataset

// COMMAND ----------

/* Now we can write customized SQL queries from the entire Dataset for exploration analysis */
DF_features.createOrReplaceTempView("Complete_Data")

// COMMAND ----------

// MAGIC %md 1. Explore the Mean Data

// COMMAND ----------

val Explore_mean_data = spark.sql("Select diagnosis, radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean from Complete_Data")

// COMMAND ----------

/* Exploratory Analysis*/
display(Explore_mean_data)

// COMMAND ----------

// MAGIC %md Explore SE Data

// COMMAND ----------

val Explore_se_data = spark.sql("Select radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se from Complete_Data")

// COMMAND ----------

/* Explore SE Data */
display(Explore_se_data)

// COMMAND ----------

// MAGIC %md Exploratory Analysis for worse cases

// COMMAND ----------

val Explore_worst_data = spark.sql("Select radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst from Complete_Data")

// COMMAND ----------

/* Explore worse case Data */
display(Explore_worst_data)

// COMMAND ----------

// MAGIC %md Advanced Level Exploratory Data Analysis for Benign Tumor

// COMMAND ----------

val DF_Benign = DF_features.filter(row => row.getAs[String]("diagnosis").contains("B") )

// COMMAND ----------

display(DF_Benign)

// COMMAND ----------

/* Now we can write customized SQL queries from Malignant Dataset for exploration analysis */
DF_Benign.createOrReplaceTempView("Non_Cancer_Data")

// COMMAND ----------

/* Create a Dynamic Query to select a record of one using id patient and analyze the kind of tunor */

val identifier = 85713702 /* you can change it to any other patient id */

/* Select all the mean records based on the id */
val radius_mean = spark.sql("Select radius_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_mean = spark.sql("Select texture_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_mean = spark.sql("Select perimeter_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_mean = spark.sql("Select area_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_mean = spark.sql("Select smoothness_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_mean = spark.sql("Select compactness_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_mean = spark.sql("Select concavity_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_mean = spark.sql("Select concave_points_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_mean = spark.sql("Select symmetry_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_mean = spark.sql("Select fractal_dimension_mean from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Select all the worst records based on the id */
val radius_worst = spark.sql("Select radius_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_worst = spark.sql("Select texture_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_worst = spark.sql("Select perimeter_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_worst = spark.sql("Select area_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_worst = spark.sql("Select smoothness_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_worst = spark.sql("Select compactness_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_worst = spark.sql("Select concavity_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_worst = spark.sql("Select concave_points_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_worst = spark.sql("Select symmetry_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_worst = spark.sql("Select fractal_dimension_worst from Non_Cancer_Data where id=" + identifier).collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Collect all average values for different means parameters */
val avg_radius = spark.sql("Select avg(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val avg_texture = spark.sql("Select avg(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_perimeter = spark.sql("Select avg(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_area = spark.sql("Select avg(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_smoothness = spark.sql("Select avg(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_compactness = spark.sql("Select avg(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concavity = spark.sql("Select avg(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concave_points = spark.sql("Select avg(concave_points_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_symmetry = spark.sql("Select avg(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_fractal_dimension = spark.sql("Select avg(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Collect all average values for different worse parameters */
val avg_radius_worst = spark.sql("Select avg(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val avg_texture_worst = spark.sql("Select avg(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_perimeter_worst = spark.sql("Select avg(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_area_worst = spark.sql("Select avg(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_smoothness_worst = spark.sql("Select avg(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_compactness_worst = spark.sql("Select avg(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concavity_worst = spark.sql("Select avg(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_concave_points_worst = spark.sql("Select avg(concave_points_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_symmetry_worst = spark.sql("Select avg(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val avg_fractal_dimension_worst = spark.sql("Select avg(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Collect all min values for different mean parameters */
val min_radius = spark.sql("Select min(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture = spark.sql("Select min(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter = spark.sql("Select min(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area = spark.sql("Select min(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness = spark.sql("Select min(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness = spark.sql("Select min(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity = spark.sql("Select min(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points = spark.sql("Select min(concave_points_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry = spark.sql("Select min(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension = spark.sql("Select min(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Collect all min values for different worst parameters */
val min_radius_worst = spark.sql("Select min(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_worst = spark.sql("Select min(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_worst = spark.sql("Select min(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_worst = spark.sql("Select min(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_worst = spark.sql("Select min(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_worst = spark.sql("Select min(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_worst = spark.sql("Select min(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_worst = spark.sql("Select min(concave_points_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_worst = spark.sql("Select min(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_worst = spark.sql("Select min(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Collect All max values for different mean parameters */
val max_radius = spark.sql("Select max(radius_mean) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture = spark.sql("Select max(texture_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter = spark.sql("Select max(perimeter_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area = spark.sql("Select max(area_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness = spark.sql("Select max(smoothness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness = spark.sql("Select max(compactness_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity = spark.sql("Select max(concavity_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points = spark.sql("Select max(concave_points_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry = spark.sql("Select max(symmetry_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension = spark.sql("Select max(fractal_dimension_mean) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Collect All max values for different worst parameters */
val max_radius_worst = spark.sql("Select max(radius_worst) from Non_Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_worst = spark.sql("Select max(texture_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_worst = spark.sql("Select max(perimeter_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_worst = spark.sql("Select max(area_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_worst = spark.sql("Select max(smoothness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_worst = spark.sql("Select max(compactness_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_worst = spark.sql("Select max(concavity_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_worst = spark.sql("Select max(concave_points_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_worst = spark.sql("Select max(symmetry_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_worst = spark.sql("Select max(fractal_dimension_worst) from Non_Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

def Calculation(x:Double, min:Double, max:Double ) : Double = {
      var Calc:Double = 0
      Calc = (x - min)/(max - min)

      return Calc
   }

// COMMAND ----------

/* Calculated mean values */
val actual_Radius = Calculation (radius_mean,min_radius,max_radius)
val actual_texture = Calculation (texture_mean,min_texture,max_texture)
val actual_perimeter = Calculation (perimeter_mean,min_perimeter,max_perimeter)
val actual_area = Calculation (area_mean,min_area,max_area)
val actual_smoothness = Calculation (smoothness_mean,min_smoothness,max_smoothness)
val actual_compactness= Calculation (compactness_mean,min_compactness,max_compactness)
val actual_concavity = Calculation (concavity_mean,min_concavity,max_concavity)
val actual_concave_points = Calculation (concave_points_mean,min_concave_points,max_concave_points)
val actual_symmetry = Calculation (symmetry_mean,min_symmetry,max_symmetry)
val actual_fractal_dimension = Calculation (fractal_dimension_mean,min_fractal_dimension,max_fractal_dimension)

/* Calculated worst values */
val actual_Radius_worst = Calculation (radius_worst,min_radius_worst,max_radius_worst)
val actual_texture_worst = Calculation (texture_worst,min_texture_worst,max_texture_worst)
val actual_perimeter_worst = Calculation (perimeter_worst,min_perimeter_worst,max_perimeter_worst)
val actual_area_worst = Calculation (area_worst,min_area_worst,max_area_worst)
val actual_smoothness_worst = Calculation (smoothness_worst,min_smoothness_worst,max_smoothness_worst)
val actual_compactness_worst= Calculation (compactness_worst,min_compactness_worst,max_compactness_worst)
val actual_concavity_worst = Calculation (concavity_worst,min_concavity_worst,max_concavity_worst)
val actual_concave_points_worst = Calculation (concave_points_worst,min_concave_points_worst,max_concave_points_worst)
val actual_symmetry_worst = Calculation (symmetry_worst,min_symmetry_worst,max_symmetry_worst)
val actual_fractal_dimension_worst = Calculation (fractal_dimension_worst,min_fractal_dimension_worst,max_fractal_dimension_worst)

// COMMAND ----------

/* Optimum will be used twice to compare both Malignant Cancer and Benign Tumor - mean values*/
val optimum_Radius = Calculation (avg_radius,min_radius,max_radius)
val optimum_texture = Calculation (avg_texture,min_texture,max_texture)
val optimum_perimeter = Calculation (avg_perimeter,min_perimeter,max_perimeter)
val optimum_area = Calculation (avg_area,min_area,max_area)
val optimum_smoothness = Calculation (avg_smoothness,min_smoothness,max_smoothness)
val optimum_compactness= Calculation (avg_compactness,min_compactness,max_compactness)
val optimum_concavity = Calculation (avg_concavity,min_concavity,max_concavity)
val optimum_concave_points = Calculation (avg_concave_points,min_concave_points,max_concave_points)
val optimum_symmetry = Calculation (avg_symmetry,min_symmetry,max_symmetry)
val optimum_fractal_dimension = Calculation (avg_fractal_dimension,min_fractal_dimension,max_fractal_dimension)

/* Optimum will be used twice to compare both Malignant Cancer and Benign Tumor - worst values*/
val optimum_Radius_worst = Calculation (avg_radius_worst,min_radius_worst,max_radius_worst)
val optimum_texture_worst = Calculation (avg_texture_worst,min_texture_worst,max_texture_worst)
val optimum_perimeter_worst = Calculation (avg_perimeter_worst,min_perimeter_worst,max_perimeter_worst)
val optimum_area_worst = Calculation (avg_area_worst,min_area_worst,max_area_worst)
val optimum_smoothness_worst = Calculation (avg_smoothness_worst,min_smoothness_worst,max_smoothness_worst)
val optimum_compactness_worst = Calculation (avg_compactness_worst,min_compactness_worst,max_compactness_worst)
val optimum_concavity_worst = Calculation (avg_concavity_worst,min_concavity_worst,max_concavity_worst)
val optimum_concave_points_worst = Calculation (avg_concave_points_worst,min_concave_points_worst,max_concave_points_worst)
val optimum_symmetry_worst = Calculation (avg_symmetry_worst,min_symmetry_worst,max_symmetry_worst)
val optimum_fractal_dimension_worst = Calculation (avg_fractal_dimension_worst,min_fractal_dimension_worst,max_fractal_dimension_worst)

// COMMAND ----------

// MAGIC %md Advanced Level Exploratory Data Analysis for Malignant Stage Cancer

// COMMAND ----------

val DF_Malignant = DF_features.filter(row => row.getAs[String]("diagnosis").contains("M") )

// COMMAND ----------

display(DF_Malignant)

// COMMAND ----------

/* Now we can write customized SQL queries from Benign Dataset for exploration analysis */
DF_Malignant.createOrReplaceTempView("Cancer_Data")

// COMMAND ----------

/* Create a Dynamic Query to select a record of one using id patient and analyze the kind of tumor - which is now a cancer type */

val identifier_m = 84300903 /* you can change it to any other patient id */

/* Select all the mean records based on the id */
val radius_mean_m = spark.sql("Select radius_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_mean_m = spark.sql("Select texture_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_mean_m = spark.sql("Select perimeter_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_mean_m = spark.sql("Select area_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_mean_m = spark.sql("Select smoothness_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_mean_m = spark.sql("Select compactness_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_mean_m = spark.sql("Select concavity_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_mean_m = spark.sql("Select concave_points_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_mean_m = spark.sql("Select symmetry_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_mean_m = spark.sql("Select fractal_dimension_mean from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Select all the worst records based on the id */
val radius_worst_m = spark.sql("Select radius_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val texture_worst_m = spark.sql("Select texture_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val perimeter_worst_m = spark.sql("Select perimeter_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val area_worst_m = spark.sql("Select area_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val smoothness_worst_m = spark.sql("Select smoothness_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val compactness_worst_m = spark.sql("Select compactness_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concavity_worst_m = spark.sql("Select concavity_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val concave_points_worst_m = spark.sql("Select concave_points_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val symmetry_worst_m = spark.sql("Select symmetry_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble
val fractal_dimension_worst_m = spark.sql("Select fractal_dimension_worst from Cancer_Data where id=" + identifier_m).collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Collect all min values for mean attributes*/
val min_radius_m = spark.sql("Select min(radius_mean) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_m = spark.sql("Select min(texture_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_m = spark.sql("Select min(perimeter_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_m = spark.sql("Select min(area_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_m = spark.sql("Select min(smoothness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_m = spark.sql("Select min(compactness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_m = spark.sql("Select min(concavity_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_m = spark.sql("Select min(concave_points_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_m = spark.sql("Select min(symmetry_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_m = spark.sql("Select min(fractal_dimension_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Collect all min values for worst attributes*/
val min_radius_worst_m = spark.sql("Select min(radius_worst) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val min_texture_worst_m = spark.sql("Select min(texture_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_perimeter_worst_m = spark.sql("Select min(perimeter_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_area_worst_m = spark.sql("Select min(area_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_smoothness_worst_m = spark.sql("Select min(smoothness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_compactness_worst_m = spark.sql("Select min(compactness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concavity_worst_m = spark.sql("Select min(concavity_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_concave_points_worst_m = spark.sql("Select min(concave_points_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_symmetry_worst_m = spark.sql("Select min(symmetry_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val min_fractal_dimension_worst_m = spark.sql("Select min(fractal_dimension_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Collect All max values for mean attributes */
val max_radius_m = spark.sql("Select max(radius_mean) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_m = spark.sql("Select max(texture_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_m = spark.sql("Select max(perimeter_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_m = spark.sql("Select max(area_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_m = spark.sql("Select max(smoothness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_m = spark.sql("Select max(compactness_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_m = spark.sql("Select max(concavity_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_m = spark.sql("Select max(concave_points_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_m = spark.sql("Select max(symmetry_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_m = spark.sql("Select max(fractal_dimension_mean) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

/* Collect all max values for worst attributes*/
val max_radius_worst_m = spark.sql("Select max(radius_worst) from Cancer_Data").collect().map(_.getDouble(0)).mkString(" ").toDouble
val max_texture_worst_m = spark.sql("Select max(texture_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_perimeter_worst_m = spark.sql("Select max(perimeter_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_area_worst_m = spark.sql("Select max(area_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_smoothness_worst_m = spark.sql("Select max(smoothness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_compactness_worst_m = spark.sql("Select max(compactness_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concavity_worst_m = spark.sql("Select max(concavity_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_concave_points_worst_m = spark.sql("Select max(concave_points_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_symmetry_worst_m = spark.sql("Select max(symmetry_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble
val max_fractal_dimension_worst_m = spark.sql("Select max(fractal_dimension_worst) from Cancer_Data").collect.map(_.getDouble(0)).mkString(" ").toDouble

// COMMAND ----------

/* Actual Malignant Calculation for mean attribute*/
val actual_Radius_m = Calculation (radius_mean_m,min_radius_m,max_radius_m)
val actual_texture_m = Calculation (texture_mean_m,min_texture_m,max_texture_m)
val actual_perimeter_m = Calculation (perimeter_mean_m,min_perimeter_m,max_perimeter_m)
val actual_area_m = Calculation (area_mean_m,min_area_m,max_area_m)
val actual_smoothness_m = Calculation (smoothness_mean_m,min_smoothness_m,max_smoothness_m)
val actual_compactness_m = Calculation (compactness_mean_m,min_compactness_m,max_compactness_m)
val actual_concavity_m = Calculation (concavity_mean_m,min_concavity_m,max_concavity_m)
val actual_concave_points_m = Calculation (concave_points_mean_m,min_concave_points_m,max_concave_points_m)
val actual_symmetry_m = Calculation (symmetry_mean_m,min_symmetry_m,max_symmetry_m)
val actual_fractal_dimension_m = Calculation (fractal_dimension_mean_m,min_fractal_dimension_m,max_fractal_dimension_m)

/* Actual Malignant Calculation for worst attributes*/
val actual_Radius_worst_m = Calculation (radius_worst_m,min_radius_worst_m,max_radius_worst_m)
val actual_texture_worst_m = Calculation (texture_worst_m,min_texture_worst_m,max_texture_worst_m)
val actual_perimeter_worst_m = Calculation (perimeter_worst_m,min_perimeter_worst_m,max_perimeter_worst_m)
val actual_area_worst_m = Calculation (area_worst_m,min_area_worst_m,max_area_worst_m)
val actual_smoothness_worst_m = Calculation (smoothness_worst_m,min_smoothness_worst_m,max_smoothness_worst_m)
val actual_compactness_worst_m = Calculation (compactness_worst_m,min_compactness_worst_m,max_compactness_worst_m)
val actual_concavity_worst_m = Calculation (concavity_worst_m,min_concavity_worst_m,max_concavity_worst_m)
val actual_concave_points_worst_m = Calculation (concave_points_worst_m,min_concave_points_worst_m,max_concave_points_worst_m)
val actual_symmetry_worst_m = Calculation (symmetry_worst_m,min_symmetry_worst_m,max_symmetry_worst_m)
val actual_fractal_dimension_worst_m = Calculation (fractal_dimension_worst_m,min_fractal_dimension_worst_m,max_fractal_dimension_worst_m)

// COMMAND ----------

/* Radar Graph to determine optimum vs actual cell type for worst attributes. */

val rs_w1 = Seq(optimum_fractal_dimension_worst, optimum_texture_worst, optimum_perimeter_worst, optimum_area_worst, optimum_smoothness_worst,                                              optimum_concavity_worst, optimum_concave_points_worst, optimum_symmetry_worst,optimum_compactness_worst,optimum_Radius_worst )

val rs_w2 = Seq(actual_fractal_dimension_worst, actual_texture_worst, actual_perimeter_worst, actual_area_worst, actual_smoothness_worst,actual_concavity_worst, actual_concave_points_worst, actual_symmetry_worst,actual_compactness_worst,actual_Radius_worst )

val rs_w3 = Seq(actual_fractal_dimension_worst_m, actual_texture_worst_m, actual_perimeter_worst_m, actual_area_worst_m, actual_smoothness_worst_m,                                              actual_concavity_worst_m, actual_concave_points_worst_m, actual_symmetry_worst_m, actual_compactness_worst_m, actual_Radius_worst_m )

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               <div style="width: 1100px;">
               <div id="myDiv_1" style="float:left;width:500px;"></div>
               <div id="myDiv_blank" style="float:left;width:100px;"></div>
               <div id="myDiv_2" style="float:left;width:500px;"></div>
               </div>
               <script>
                      data_1 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_w1(0)}, ${rs_w1(1)}, ${rs_w1(2)}, ${rs_w1(3)}, ${rs_w1(4)}, ${rs_w1(5)}, ${rs_w1(6)}, ${rs_w1(7)}, ${rs_w1(8)}, ${rs_w1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_w2(0)}, ${rs_w2(1)}, ${rs_w2(2)}, ${rs_w2(3)}, ${rs_w2(4)}, ${rs_w2(5)}, ${rs_w2(6)}, ${rs_w2(7)}, ${rs_w2(8)}, ${rs_w2(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Benign Tumor Patient'
                          }
                        ]
                      
                      data_2 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_w1(0)}, ${rs_w1(1)}, ${rs_w1(2)}, ${rs_w1(3)}, ${rs_w1(4)}, ${rs_w1(5)}, ${rs_w1(6)}, ${rs_w1(7)}, ${rs_w1(8)}, ${rs_w1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_w3(0)}, ${rs_w3(1)}, ${rs_w3(2)}, ${rs_w3(3)}, ${rs_w3(4)}, ${rs_w3(5)}, ${rs_w3(6)}, ${rs_w3(7)}, ${rs_w3(8)}, ${rs_w3(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Malignant Cancer Patient'
                          }
                        ]
                        
                      layout_1 = {
                        title : "Analyzing the worst across different dimensions for Benign Tumor",
                        "titlefont": {
                                              family : 'Verdana', size:12, color:'#7f7f7f'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }
                        
                        layout_2 = {
                        title : "Analyzing the worst across different dimensions for Malignant Cancer",
                        "titlefont": {
                                              family : 'Verdana', size:12, color:'#7f7f7f'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }

                        Plotly.plot("myDiv_1", data_1, layout_1)
                        Plotly.plot("myDiv_2", data_2, layout_2)
               </script>
               
</head> """)

// COMMAND ----------

/* Radar Graph to determine optimum vs actual cell type for means. */

val rs_1 = Seq(optimum_fractal_dimension, optimum_texture, optimum_perimeter, optimum_area, optimum_smoothness,                                              optimum_concavity, optimum_concave_points, optimum_symmetry,optimum_compactness,optimum_Radius )

val rs_2 = Seq(actual_fractal_dimension, actual_texture, actual_perimeter, actual_area, actual_smoothness, actual_concavity, actual_concave_points, actual_symmetry,actual_compactness,actual_Radius )

val rs_3 = Seq(actual_fractal_dimension_m, actual_texture_m, actual_perimeter_m, actual_area_m, actual_smoothness_m,                                              actual_concavity_m, actual_concave_points_m, actual_symmetry_m, actual_compactness_m, actual_Radius_m )

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               <div style="width: 1100px;">
               <div id="myDiv_1" style="float:left;width:500px;"></div>
               <div id="myDiv_blank" style="float:left;width:100px;"></div>
               <div id="myDiv_2" style="float:left;width:500px;"></div>
               </div>
               <script>
                      data_1 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_1(0)}, ${rs_1(1)}, ${rs_1(2)}, ${rs_1(3)}, ${rs_1(4)}, ${rs_1(5)}, ${rs_1(6)}, ${rs_1(7)}, ${rs_1(8)}, ${rs_1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_2(0)}, ${rs_2(1)}, ${rs_2(2)}, ${rs_2(3)}, ${rs_2(4)}, ${rs_2(5)}, ${rs_2(6)}, ${rs_2(7)}, ${rs_2(8)}, ${rs_2(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Benign Tumor Patient'
                          }
                        ]
                      
                      data_2 = [
                       {
                       type: 'scatterpolar',
                       r: [${rs_1(0)}, ${rs_1(1)}, ${rs_1(2)}, ${rs_1(3)}, ${rs_1(4)}, ${rs_1(5)}, ${rs_1(6)}, ${rs_1(7)}, ${rs_1(8)}, ${rs_1(9)}],
                       theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness',  'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Average size of Tumor'
                          
                          },
                          {
                          type: 'scatterpolar',
                          r: [${rs_3(0)}, ${rs_3(1)}, ${rs_3(2)}, ${rs_3(3)}, ${rs_3(4)}, ${rs_3(5)}, ${rs_3(6)}, ${rs_3(7)}, ${rs_3(8)}, ${rs_3(9)}],
                          theta: ['fractal_dimension','texture','perimeter','area','smoothness', 'concavity','concave_points' ,'symmetry', 'compactness', 'Radius'],
                          
                          fill: 'toself',
                          
                          name: 'Malignant Cancer Patient'
                          }
                        ]
                        
                      layout_1 = {
                        title : "Analyzing the means across different dimensions for Benign Tumor",
                        "titlefont": {
                                              family : 'Verdana', size:12, color:'#7f7f7f'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }
                        
                        layout_2 = {
                        
                        title : "Analyzing the means across different dimensions for Malignant Cancer",
                        "titlefont": {
                                              family : 'Verdana', size:12, color:'#7f7f7f'
                                     }, 
                          polar: {
                        radialaxis: {
      
                              visible: false,
                              range: [0, 1]
                            }
                          }
                        }

                        Plotly.plot("myDiv_1", data_1, layout_1)
                        Plotly.plot("myDiv_2", data_2, layout_2)
               </script>
               
</head> """)

// COMMAND ----------

// MAGIC %md Identify the features and labels

// COMMAND ----------

val nonFeatureCols = Array("id","diagnosis","Int_Diagnostics")
val features = DF_features.columns.diff(nonFeatureCols)

// COMMAND ----------

val Array(training, test) = DF_features.randomSplit(Array(0.8, 0.2),seed = 12345)

// Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

// COMMAND ----------

val assembler = new VectorAssembler().setInputCols(features).setOutputCol("Resultfeatures")

val FeaturesPipeline = (new Pipeline()
  .setStages(Array(assembler)))

val trainingFit = FeaturesPipeline.fit(training)
val trainingFeatures = trainingFit.transform(training)
val testFeatures = trainingFit.transform(test)

// COMMAND ----------

// Add a standard scaler to scale the features before applying PCA
import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
  .setInputCol("Resultfeatures")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false)

val scaler_fit = scaler.fit(trainingFeatures)

val scaler_training = scaler_fit.transform(trainingFeatures)
val scaler_test = scaler_fit.transform(testFeatures)

// COMMAND ----------

// Include PCA

val pca = new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(2)
  .fit(scaler_training)

val pca_training = pca.transform(scaler_training)
val pca_test = pca.transform(scaler_test)
val Int_Diagnostics = pca_training.select("Int_Diagnostics")

// COMMAND ----------

/* Show the PCA Results - How PCA fairs on the training set*/
val  pca_results= pca.transform(scaler_training).select("pcaFeatures","Int_Diagnostics")

// COMMAND ----------

pca_results.show(false)

// COMMAND ----------

/* We now transform the vector to a dataframe */
import org.apache.spark.sql.functions._
import org.apache.spark.ml._

val df = Seq( (1 , linalg.Vectors.dense(1,0) ) ).toDF("id", "features")
// A UDF to convert VectorUDT to ArrayType
val vecToArray = udf( (xs: linalg.Vector) => xs.toArray)

// Add a ArrayType Column   
val dfArr = pca_results.withColumn("Features" , vecToArray($"pcaFeatures"))

// COMMAND ----------

// Create an array for the column names
val pca_elements = Array("PCA_1", "PCA_2")

// Create a SQL-like expression using the array 
val sqlExpr = pca_elements.zipWithIndex.map{ case (alias, idx) => col("Features").getItem(idx).as(alias) }

// Extract Elements from dfArr and put it inside DataFrame    
val pcaDF = dfArr.select(sqlExpr : _*).toDF

val newPCADF = dfArr.select("Int_Diagnostics")

// COMMAND ----------

//Now create a temp Id to perform join operation

import org.apache.spark.sql.types._

val Temp_1 = spark.sqlContext.createDataFrame(
  newPCADF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(newPCADF.schema.fields :+ StructField("index", LongType, false))
)

val Temp_2 = spark.sqlContext.createDataFrame(
  pcaDF.rdd.zipWithIndex.map {
    case (row, index) => Row.fromSeq(row.toSeq :+ index)
  },
  // Create schema for index column
  StructType(pcaDF.schema.fields :+ StructField("index", LongType, false))
)

// COMMAND ----------

val final_PCA = Temp_2.join(Temp_1, Seq("index")).drop("index")

// COMMAND ----------

display(final_PCA)

// COMMAND ----------

// MAGIC %md  After Scaling, PCA does quite a decent job of visualising our two target clusters ( 1 for Malignant and 0 for Benign)

// COMMAND ----------

// MAGIC %md Although PCA is able to differentiate the classes very well, I found later that PCA technique also reduced some important features - which resulted in reduced accuracy. So i will only use standard scaler on the dataset.

// COMMAND ----------

// MAGIC %md Build The Models to predict the tumor type. Build Random Forest, Support Vector Machine, Logistic Regression, Light GBM and XG BOOST Classifier

// COMMAND ----------

// Now that the data has been prepared, let's split the training dataset into a training and validation dataframe
val Array(trainDF, valDF) = pca_training.randomSplit(Array(0.8, 0.2),seed = 12345)

// COMMAND ----------

// Create default param map for XGBoost
def get_param_xgb(): mutable.HashMap[String, Any] = {
    val params = new mutable.HashMap[String, Any]()
        params += "eta" -> 0.3
        params += "max_depth" -> 6
        params += "gamma" -> 0.0
        params += "colsample_bylevel" -> 1
        params += "objective" -> "binary:logistic"
        params += "num_class" -> 2
        params += "booster" -> "gbtree"
        params += "num_rounds" -> 1
        params += "nWorkers" -> 1
    return params
}

// COMMAND ----------

// Create XGBoost Classifier, Random Forest Classifier, Light GBM and SVC
val xgb_model = new XGBoostEstimator(get_param_xgb().toMap).setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val rnf_model = new RandomForestClassifier().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val lgbm_model = new LightGBMClassifier().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val svc_model = new LinearSVC().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")
val log_model = new LogisticRegression().setLabelCol("Int_Diagnostics").setFeaturesCol("scaledFeatures")

// COMMAND ----------

// Setup the binary classifier evaluator
val evaluator_binary = (new BinaryClassificationEvaluator()
  .setLabelCol("Int_Diagnostics")
  .setRawPredictionCol("Prediction")
  .setMetricName("areaUnderROC"))

// COMMAND ----------

// MAGIC %md XG BOOST Model

// COMMAND ----------

// Fit the model on XGBOOST
val fit_xgb = xgb_model.fit(trainDF)
val train_pred_xgb = fit_xgb.transform(trainDF).selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//XG BOOST - Training set
evaluator_binary.evaluate(train_pred_xgb)

// COMMAND ----------

// Print the XGB Model Parameters
println("Printing out the model Parameters:")
println(xgb_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on validation set
val holdout_xgb = fit_xgb
  .transform(valDF)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_xgb)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_xgb = fit_xgb
  .transform(pca_test)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_xgb)

// COMMAND ----------

// MAGIC %md Random Forest Model

// COMMAND ----------

// Fit the model on Random Forest
val fit_rnf = rnf_model.fit(trainDF)
val train_pred_rnf = fit_rnf.transform(trainDF).selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//Random Forest - Training set
evaluator_binary.evaluate(train_pred_rnf)

// COMMAND ----------

// Print the Random Forest Model Parameters
println("Printing out the model Parameters:")
println(rnf_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on validation set
val holdout_rnf = fit_rnf
  .transform(valDF)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_rnf)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_rnf = fit_rnf
  .transform(pca_test)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_rnf)

// COMMAND ----------

// MAGIC %md Light GBM Model

// COMMAND ----------

// Fit the model on Light GBM Forest
val fit_lgbm = lgbm_model.fit(trainDF)
val train_pred_lgbm = fit_lgbm.transform(trainDF).selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

// Light GBM - Training set
evaluator_binary.evaluate(train_pred_lgbm)

// COMMAND ----------

// Print the Light GBM Model Parameters
println("Printing out the model Parameters:")
println(lgbm_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on validation set
val holdout_lgbm = fit_lgbm
  .transform(valDF)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_lgbm)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_lgbm = fit_lgbm
  .transform(pca_test)
  .selectExpr("Prediction", "Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_lgbm)

// COMMAND ----------

// MAGIC %md Support Vector Classification Model

// COMMAND ----------

// Fit the model on SVC
val fit_svc = svc_model.fit(trainDF)
val train_pred_svc = fit_svc.transform(trainDF).selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//SVC Model - Training set
evaluator_binary.evaluate(train_pred_svc)

// COMMAND ----------

// Print the SVC Model Parameters
println("Printing out the model Parameters:")
println(svc_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on validation set
val holdout_svc = fit_svc
  .transform(valDF)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_svc)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_svc = fit_svc
  .transform(pca_test)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_svc)

// COMMAND ----------

// MAGIC %md and finally the Logistic Regression Model

// COMMAND ----------

// Fit the model on logistic regression model
val fit_log = log_model.fit(trainDF)
val train_pred_log = fit_log.transform(trainDF).selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

//logistic regression Model - Training set
evaluator_binary.evaluate(train_pred_svc)

// COMMAND ----------

// Print the Logistic regression model Parameters
println("Printing out the model Parameters:")
println(log_model.explainParams)
println("-"*20)

// COMMAND ----------

// Now check the accuracy on validation set
val holdout_log = fit_log
  .transform(valDF)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_log)

// COMMAND ----------

// Accuracy on Test Set
val holdout_test_log = fit_log
  .transform(pca_test)
  .selectExpr("Prediction", "cast(Int_Diagnostics as Double) Int_Diagnostics",
    """CASE Prediction = Int_Diagnostics
  WHEN true then 1
  ELSE 0
END as equal""")

evaluator_binary.evaluate(holdout_test_log)

// COMMAND ----------

// MAGIC %md Lets Summarize the results for these 5 models, in terms of accuracy on the test set

// COMMAND ----------

val test_svc = evaluator_binary.evaluate(holdout_test_svc)
val test_rnf = evaluator_binary.evaluate(holdout_test_rnf)
val test_log = evaluator_binary.evaluate(holdout_test_log)
val test_lgbm = evaluator_binary.evaluate(holdout_test_lgbm)
val test_xgb = evaluator_binary.evaluate(holdout_test_xgb)

val Xs = Seq(test_xgb,test_rnf,test_svc,test_log,test_lgbm)

displayHTML(s""" <head>
               <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
               
               <div id="tester" style="width:400px;height:400px;"></div>
               
               <script>
                 TESTER = document.getElementById('tester');
                 
                var trace1 = {
                                x: ['XG Boost', 'Random Forest', 'SVC', 'Logistic', 'Light GBM'],
                                y: [${Xs(0)}, ${Xs(1)}, ${Xs(2)}, ${Xs(3)}, ${Xs(4)}],
                                marker:{
                                  color: ['rgba(204,204,204,1)', 'rgba(204,204,204,1)', 'rgba(222,45,38,0.8)','rgba(204,204,204,1)', 'rgba(204,204,204,1)']
                                },
                                type: 'bar'
                              };
             
                              
                              var data = [trace1];

                              var layout = {
                                title: 'Model Comparison Measure : Accuracy',
                                size: 12,
                                   yaxis: {
                                title: 'Accuracy on the test set',
                                range: [.85,.99],                                      
                                  titlefont: {
                                  size: 12,
                                  color: '#7f7f7f'
                                }
                              }
                              };
                 Plotly.newPlot(TESTER, data, layout);
                </script>
               
</head> """)

// COMMAND ----------

// MAGIC %md <pre>
// MAGIC <B>Compute Other Classification Metrics for Support Vector Classifier Model</b>
// MAGIC 
// MAGIC True positives are how often the model correctly predicted a tumour was malignant
// MAGIC False positives are how often the model predicted a tumour was malignant when it was benign
// MAGIC False negatives indicate how the model correctly predicted a tumour was benign
// MAGIC False Positive indicate how often the model predicted a tumour was benign when in fact it was malignant
// MAGIC 
// MAGIC </pre>

// COMMAND ----------

// logistic regression model - Training Set

val lp_svc = train_pred_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = train_pred_svc.count()

// True Positives + False Negatives
val correct = lp_svc.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_svc.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_svc.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_svc.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_svc.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_svc.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)

// COMMAND ----------

//Get the area under the curve
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
val predictionLabels = train_pred_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC

// COMMAND ----------

// logistic regression model - validation Set

val lp_svc = holdout_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = holdout_svc.count()

// True Positives + False Negatives
val correct = lp_svc.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_svc.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_svc.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_svc.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_svc.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_svc.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)

// COMMAND ----------

// ROC for Vaidation Set - Logistic Regression

val predictionLabels = holdout_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC

// COMMAND ----------

// logistic regression model - Test Set

val lp_log = holdout_test_svc.select("Prediction", "Int_Diagnostics")

//Total Records
val counttotal = holdout_test_svc.count()

// True Positives + False Negatives
val correct = lp_log.filter($"Int_Diagnostics" === $"Prediction").count()

// True Negatives + False Positives
val wrong = lp_log.filter(not($"Int_Diagnostics" === $"Prediction")).count()

//True Positive
val truep = lp_log.filter($"Prediction" === 1.0).filter($"Int_Diagnostics" === $"Prediction").count()

//True Negative
val falseN = lp_log.filter($"Prediction" === 0.0).filter($"Int_Diagnostics" === $"Prediction").count()

// False Negative
val falseP = lp_log.filter($"Prediction" === 0.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//False Positive
val truen = lp_log.filter($"Prediction" === 1.0).filter(not($"Int_Diagnostics" === $"Prediction")).count()

//Precision
val Precision = truep.toDouble / (truep.toDouble + falseP.toDouble)

//Recall
val Recall = truep.toDouble / (truep.toDouble + falseN.toDouble)

// COMMAND ----------

// ROC for Vaidation Set - SVC Classification Model

val predictionLabels = holdout_test_svc.select("Prediction", "Int_Diagnostics").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val binMetrics = new BinaryClassificationMetrics(predictionLabels)

val roc_log_train = binMetrics.areaUnderROC

// COMMAND ----------

// MAGIC %md Conclusion : Support Vector Classifier is the best model based on accuracy measure : ROCAUC = 99.36%
