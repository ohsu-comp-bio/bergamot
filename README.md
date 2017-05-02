SPENCE
======

Library of automated machine learning methods and tools for systems biology analysis.

With code from
 - jeenalee
 - jdurbin
 - prismofeverything
 - kellrott


SciKit Learn methods


# sklearn_json
A python library to convert json represented ML model into scikit-learn classifier.

This library is based on json represented [machine learning model protobuf schema] (https://github.com/kellrott/ml-schema/blob/master/proto/ml_schema.proto).
It takes a json file as a parameter, and returns a scikit learn classifier. It supports Linear SVC, Linear Regression, and Logistic Regression.

### Usage
```
import sklearn_json
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.from_json('path/to/json')

lr.predict(data_to_apply_model)
```


![build-status](https://travis-ci.org/bmeg/make_prediction.svg)
# Description
This is a Python library (work in progress) for making predictions based on existing linear models. Currently, it supports logistic regression models that predict binary states.

# Install
`pip install git+https://github.com/bmeg/make_prediction.git`

# Usage
```
import make_prediction
logistic_model = make_prediction.LogisticRegression([coef], intercept)
prediction = logistic_model(data)
```
# TODO
#### misc
- how to handle floating points?

#### test
- write test for logistic regression `predict_proba`
- write test for logistic regression `decision_function`
- ~~set up Travis-CI~~

#### src
- write `log_proba` function
- write linear regression




# normalization
Code for various signature normalization schemes.  This project will contain code for two or more normalization schemes in several languages.  

## exponential
Exponential normalization is a rank based normalization scheme.  Genes are ranked and an inverse exponential transformation is applied to each rank value to get a transformed value.  The net effect is that whatever the distribution of the input, the output is guaranteed to have an exponential distribution. 

The java version of the code can be used like this: 

```java
  import bmeg.ExponentialNormalization
  outputValues = ExponentialNormalization.transform(inputValues)
```

To build the jar file 
```
  cd exponential/java
  mvn package
```
  
The required Apache Math3 library will be downloaded by the pom file dependency.  You can include the dependency 
from that pom file in any project that will use this code.  

A groovy test script with test input and reference output is included in exponential/testscript/

## quantile

Quantile normalization maps one empirical distribtuion onto another. It is used on batches of samples to make all of the samples have the same distribution.  I can also be used to transform test samples into the distribution of training samples.  Many implementations of this for expression data make the simplifying assumption that all of the samples being normalized have the same number of genes.   Most also assume that all the data is available at the same time.   For signature normalization neither of these assumptions typically hold.   We may train the classifiers on historical data with only 9000 genes, or on recent data with 25,000 genes.  Similarly, we may want to apply the signatures to data with an unknown number of genes.  More importantly, unless we want all of the training data to travel with the models, we will not have all of the training data in hand when we wish to apply normalization to a new test sample.  We will just have some kind of summary description of the original distribution. 

Approaches:

1. We could save the entire training distribution.  This is exceedingly space inefficient, making the entire
	training dataset travel around with the models. 
2. We could artificially force samples to have the same number of genes by filling in missing genes with NaN or if the size mismatch goes the other way, subsampling genes and just save one value per gene.  This is very ad hoc.  
3. We can represent the distribution in a compact but robust way with reasonable error bounds. 

Option 3 is easy to do with existing libraries.  For example, [The COLT QuantileBin1D](https://dst.lbl.gov/ACSSoftware/colt/api/hep/aida/bin/QuantileBin1D.html) class efficiently stores a compressed version of a distribution guaranteed to meet certain error bounds.   For a system purely implemented in Java I would recommend this approach.   For our application, we would like to share models across language barriers.  So we need a sparse representation of the distribution that we can save as a protocolbuffer.  The simplest thing is to save a list of quantiles and exchange that.  Given, say, a 17000 genes x 100 sample training set, there are 1,700,000 values in the training
distribution.  The key question to resolve is how to shrink these values into a manageable set of no more than, say, 20,000 values. Sampling, averaging, and shrinking (saving every kth value from sorted list) are all options but short of re-implementing the guaranteed bounds shrinking/sampling scheme of QuantileBin1D, it isn't clear which simple method is best.  
 
#### TrainingSetQuantileNormalization

Implementation of training set quantile normalization designed to be simple enough to easily translate into other languages.  This implementation just uses simple sampling to extract a reasonable resolution approximation out of the full distribution. This code can also be used as-is in any JVM based pipeline.   

To train the values use it like this:

```java
  import bmeg.TrainingSetQuantileNormalization;
  // Resolution of quantile approximation (number of points to save).  
  int numQuantiles = 20000; 
  TrainingSetQuantileNormalization tqn = new TrainingSetQuantileNormalization(numQuantiles);
  qn.compressDistribution(trainingValues);
  qn.save(fileName); // saves a list of sampled values
```

To use trained values to transform new samples use it like:

```java
  import bmeg.TrainingSetQuantileNormalization;
  TrainingSetQuantileNormalization tqn = TrainingSetQuantileNormalization.read(fileName);
  transformedValues = qn.transform(inputValues);
```

To build the jar file 
```
  cd quantile
  mvn package
```

 

#### QuantileNormalizationReference

A reference version of quantile normalization that uses the QuantileBin1D class. TrainingSetQuantileNormalization and this class should produce comparable results.  This class has the advantage of guaranteed error bounds and a more robust compression of the training distribution.  Using this code has two phases.  First creating a compressed version of the distribution.  This is done like:

```java
  import bmeg.QuantileNormalizationReference;
  QuantileNormalizationReference qn = new QuantileNormalizationReference();
  qn.saveDistribution(trainingValues);
  qn.save(fileName); // saves a serialized version of the class
```
Then, when applying this trained model to new samples you can read in the saved version of the distribution
and apply it to new samples like:

```java
  import bmeg.QuantileNormalizationReference;
  QuantileNormalizationReference qn = QuantileNormalizationReference.read(fileName);
  transformedValues = qn.transform(inputValues);
```

To build the jar file 
```
  cd quantile
  mvn package
```
