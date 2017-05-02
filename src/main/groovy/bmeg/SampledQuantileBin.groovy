import cern.jet.random.engine.*
import cern.jet.random.sampling.RandomSamplingAssistant;

/**
*  A simplified version of QuantileBin1D, where the distribution is just represented as a 
*  simple list of elements and the compression is just sampling.   
*/ 
class SampledQuantileBin {
	
	// quantileValues is an ArrayList
	// these are the values that will be saved to define the quantiles. 
	def quantileValues = []
	
	// A fancy random number generator, but any will do. 
	RandomEngine generator = new MersenneTwister(new java.util.Date());
	RandomSamplingAssistant samplingAssistant;
		
	/**
	* n 	 number of elements to use in the distribution representation. 
	* values the full set of values for the distiribution
	*/ 
	def SampledQuantileBin(n,values){
		// Sampling assistant just helps pick n random elements out of N to 
		// save as the representation of the distribution.  
		samplingAssistant = new RandomSamplingAssistant(n,values.size(),generator);	
		values.each{value->
			if (samplingAssistant.sampleNextElement()){
				quantileValues.add(value)
			}
		}		
		quantileValues = quantileValues.sort()
	}		

	/***
	*	Returns value for given quantile. 
	*/ 
	def quantile(double quantile){
		int triggerPosition = quantile * quantileValues.size() 
		def value = quantileValues[triggerPosition]
		return(value)	
	}

	/**
	* Returns quantile for a given value. 
	*/ 
	def inverseQuantile(double value){
		int rank = rank(value)
		def quantile = rank/quantileValues.size()
		return(quantile)	
	}
	
	/**
	* find number of elements less than or equal to value
	*/ 
	def rank(double value){
		int rankCount = 0
		quantileValues.each{e->
			if (e <= value) rankCount++
		}
		return(rankCount)
	}
	
	def save(String fileName){
		new File(fileName).withWriter{w->
			quantileValues.each{value-> w.writeLine "${value.round(6)}"}
		}
	}
	
	// factory reader
	static def read(String fileName){
		def qvalues = new File(fileName).collect{it as double}
		def sqb = new SampledQuantileBin(qvalues.size(),qvalues)
		return(sqb)
	}	
	
}