import collection.JavaConverters._

import com.atilika.kuromoji.ipadic.{Token, Tokenizer}

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row

case class DocumentWithId(id: Long, value: List[String])

object WordCount {
  def main(args: Array[String]) {
    // Read a single text file into a single row
    val textDirectory = "./src/main/resources/textfiles/"
    val spark = SparkSession.builder.appName("WordCount").getOrCreate()
    val text = spark.read.option("wholetext", true).textFile(textDirectory)
                    .withColumn("id", monotonically_increasing_id).cache

    // Tokenize each text
    import spark.implicits._
    val words = text.map(row => DocumentWithId(row.getAs[Long]("id"), tokenize(row.getAs("value"))))

    // tf
    val hashingTF = new HashingTF().setInputCol("value").setOutputCol("tf")
    val wordFrequencies = hashingTF.transform(words)

    // idf
    val idf = new IDF().setInputCol("tf").setOutputCol("tfidf")
    val idfModel = idf.fit(wordFrequencies)
    val rescaledData = idfModel.transform(wordFrequencies)

    // The column "tfidf" has tfidf values as org.apache.spark.ml.linalg.SparseVector type.
    rescaledData.select("id", "tfidf").show

    spark.stop()
  }

  def tokenize(text: String) : List[String] = {
    val tokenizer = new Tokenizer()
    val sentences = text.split('\n')
    sentences.map(s => tokenizer.tokenize(s).asScala)
             .flatMap(tokens => tokens.map(t => t.getSurface))
             .toList
  }
}
