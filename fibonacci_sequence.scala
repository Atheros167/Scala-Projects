//%md
//  ## Second Scala Code: Fibonacci Sequence
//
//  As an experimental project,
//1. learning to work with integers and
//  2. Creating Objects with Main and
//  3. Creating Objects without Main


import org.apache.spark.sql.SparkSession
import scala.collection.mutable.ListBuffer
import scala.io.StdIn.readLine
import scala.collection.mutable.ArrayBuffer

// Fibonacci Sequence through a Object with Main Function
object Fibonacci {
  def main(n: Int) {
    val fib =new ListBuffer[Int]()
    var (a:Int,b:Int)=(0,1)
    fib += a
    fib += b
    var x:Int=1
    while (x < n-1) {
      var a_0=b
      b=b+a
      a=a_0
      fib += b
      x=x+1

    }
    println(n+"th element of the fibonacci sequence is "+fib(n-1))
    println("The complete fibonacci sequence upto the "+n+"th element is "+fib)
  }
}

Fibonacci.main(10)


// Fibonacci Sequence through a Object without Main Function

object Fibonacci2 extends App {
  var y: Int =0
  while (y <= (args.length-1)) {
    val fib =new ListBuffer[Int]()
    var (a:Int,b:Int)=(0,1)
    fib += a
    fib += b
    var x: Int =1

    while (x < (args(y).toInt-1)) {
      var a_0=b
      b=b+a
      a=a_0
      fib += b
      x+=1
    }

    println(args(y)+"th element of the fibonacci sequence is "+fib(args(y).toInt-1))
    println("The complete fibonacci sequence upto the "+args(y)+"th element is "+fib)
    y+=1
  }
}


Fibonacci2.main(Array("10","9","7"))