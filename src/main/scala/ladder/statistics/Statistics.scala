package ladder.statistics

import breeze.linalg._
import breeze.stats._
import scala.math.sqrt

import ladder.transform.ExponentialNormalization

object Statistics {
  def exponentialNormalization(levels: Map[String, Double]): Map[String, Double] = {
    val pairs = levels.toArray
    val keys = pairs.map(_._1)
    val coefficients = pairs.map(_._2)
    val normalized = ExponentialNormalization.transform(coefficients)
    keys.zip(normalized).toMap
  }

  def pearson(a: SparseVector[Double], b: SparseVector[Double]): Double = {
    if (a.length != b.length)
      throw new IllegalArgumentException("Vectors not of the same length.")

    val n = a.length

    val dot = a.dot(b)
    val adot = a.dot(a)
    val bdot = b.dot(b)
    val amean = mean(a)
    val bmean = mean(b)

    (dot - n * amean * bmean) / (sqrt(adot - n * amean * amean) * sqrt(bdot - n * bmean * bmean))
  }

  def pearson(a: Vector[Double], b: Vector[Double]): Double = {
    if (a.isInstanceOf[SparseVector[Double]] && b.isInstanceOf[SparseVector[Double]]) {
      return pearson(a.asInstanceOf[SparseVector[Double]], b.asInstanceOf[SparseVector[Double]])
    }

    if (a.length != b.length)
      throw new IllegalArgumentException("Vectors not of the same length.")

    val n = a.length

    val dot = a.dot(b)
    val adot = a.dot(a)
    val bdot = b.dot(b)
    val amean = mean(a)
    val bmean = mean(b)

    (dot - n * amean * bmean) / (sqrt(adot - n * amean * amean) * sqrt(bdot - n * bmean * bmean))
  }
}
