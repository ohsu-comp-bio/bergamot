package ladder.logistic

import scala.math._
import ladder.math._

class LogisticRegression(coef: Vector[Double], intercept: Double) {
  def decisionFunction(data: Vector[Vector[Double]]): Vector[Double] = {
    Math.applyDot(coef) (intercept) (data)
  }

  def prediction(data: Vector[Vector[Double]]): Vector[Double] = {
    decisionFunction(data).map(Math.logistic(_))
  }

  def logPrediction(data:Vector[Vector[Double]]): Vector[Double] = {
    prediction(data).map(p => log(p))
  }

  def predict(data: Vector[Vector[Double]], threshold: Double = 0.5): Vector[Boolean] = {
    decisionFunction(data).map(_ > threshold)
  }
}
