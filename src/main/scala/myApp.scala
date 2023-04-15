import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, Imputer, StandardScaler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}


object myApp {
def main(args : Array[String]): Unit = {
  val spark: SparkSession = SparkSession.builder()
    .master("local[*]")
    .appName("Machine Learning")
    .getOrCreate()
  var df = spark.read.format("csv").option("header", "true").load("/home/vaibhav/Downloads/framingham_heart_disease.csv");
  df = df.drop("education")
  df.show()
  df.printSchema()
  val indexers = Array("male" , "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD").map(col => new StringIndexer().setInputCol(col).setOutputCol(col+"_index"))
  val indexerPipeline = new Pipeline().setStages(indexers)
  var indexedDF = indexerPipeline.fit(df).transform(df)
  indexedDF = indexedDF.drop("male" , "age", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD")

//  replace na with 0
//  fill na values with mean in indexedDF
  
//  use vector assembler to combine all the features into a single vector column
  val assembler = new VectorAssembler()
    .setInputCols(Array("male_index" , "currentSmoker_index", "cigsPerDay_index", "BPMeds_index", "prevalentStroke_index", "prevalentHyp_index", "diabetes_index", "totChol_index", "sysBP_index", "diaBP_index", "BMI_index", "heartRate_index", "glucose_index"))
    .setOutputCol("features")
  val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithMean(true).setWithStd(true)
  val scalerPipeline = new Pipeline().setStages(Array(assembler, scaler))
  val scaledDF = scalerPipeline.fit(indexedDF).transform(indexedDF)
  scaledDF.show()
  val Array(trainDF, testDF) = scaledDF.randomSplit(Array(0.8, 0.2), seed=42)

  val lr = new LogisticRegression().setFeaturesCol("scaledFeatures").setLabelCol("TenYearCHD_index").setMaxIter(20).setRegParam(0.3).setElasticNetParam(0.8)
  val lrModel = lr.fit(trainDF)
  val predictions = lrModel.transform(testDF)

  val binaryEvaluator = new BinaryClassificationEvaluator().setRawPredictionCol("rawPrediction").setLabelCol("TenYearCHD_index")
  println(s"Test AUC: ${binaryEvaluator.evaluate(predictions)}")
}
}
