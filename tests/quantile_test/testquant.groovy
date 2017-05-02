#!/usr/bin/env groovy 

import bmeg.QuantileNormalizationReference

// Normally distributed (microarray) sample
// New sample defined over 1000 values. 
inputValues = new File("reference_input100.tab").collect{it as double}
inputValues = inputValues as double[] // ArrayList is result of collect

// Output transformed by reference pipeline..
// This distribution is defined over 9803 values...
referenceOutput = new File("reference_output.tab").collect{it as double}

qn = new QuantileNormalizationReference()
qn.saveDistribution(referenceOutput)
outputValues = qn.transform(inputValues)

outputValues.eachWithIndex{v,i->
	println "p$i\t$v"
}

/*
outputValues.eachWithIndex{v,i->
	ref = referenceOutput[i]
	delta = Math.abs(v-ref)		
	if (delta > 0.001) System.err.println "MISMATCH at val: $v\tref: $ref\tdelta: $delta"
}
*/

