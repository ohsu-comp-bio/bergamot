package ladder.math

import scala.math._

object Math {
  def logistic(t: Double): Double = {
    1.0 / (1.0 + pow(E, -t))
  }

  def dot(coefficients: Vector[Double]) (intercept: Double) (value: Vector[Double]): Double = {
    coefficients.zip(value).foldLeft(intercept) ((total, dot) => total + dot._1 * dot._2)
  }

  def applyDot(coefficients: Vector[Double]) (intercept: Double = 0) (vectors: Vector[Vector[Double]]): Vector[Double] = {
    vectors.map(dot(coefficients) (intercept))
  }
}
