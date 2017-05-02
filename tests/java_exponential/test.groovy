#!/usr/bin/env groovy 

// Grab the required library in case it's not in classpath
@Grab(group='org.apache.commons', module='commons-math3', version='3.6.1')

import bmeg.ExponentialNormalization

// Normally distributed (microarray) sample
inputValues = new File("reference_input.tab").collect{it as double}
inputValues = inputValues as double[] // ArrayList is result of collect

// Output transformed by reference pipeline..
referenceOutput = new File("reference_output.tab").collect{it as double}

outputValues = ExponentialNormalization.transform(inputValues)

outputValues.eachWithIndex{v,i->
	ref = referenceOutput[i]
	delta = Math.abs(v-ref)		
	if (delta > 0.001) System.err.println "MISMATCH at val: $v\tref: $ref\tdelta: $delta"
}

