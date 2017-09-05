import org.apache.spark.sql.SparkSession
import scala.collection.mutable.ListBuffer
import scala.io.StdIn.readLine
import scala.collection.mutable.ArrayBuffer


println("Print Output")
// First word
val input = "Yellow"

// Second Word
val check="Buffalo"

val count:Int=0
var newcount=new ListBuffer[Int]()
var common=new ArrayBuffer[String]()



for (len<-input.distinct){
  for (el <-check.distinct){
    if (len==el){
      newcount += count+1
      common = common ++ Array(len.toString)
    }
  }
}

var res:String = ("")
var x=0
while (x < newcount.sum){
  res = res ++ (common(x).toString)
  x+=x+1
}


println(newcount.sum+ " of the alphabets of the word " +"'"+ check +"'"+ " were found in " +"'"+ input+ "'")
println("The following characters are common in the two words being compared here "+ res.split("").mkString(","))