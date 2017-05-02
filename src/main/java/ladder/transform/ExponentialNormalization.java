package ladder.transform;

import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.stat.ranking.*;

public class ExponentialNormalization {
    
    static ExponentialDistribution exp = new ExponentialDistribution(1.0);

    /**
     *  Takes a single sample of data (e.g. all the genes for one patient)
     *  Returns an exponentially normalized version of that data. 
     */ 
    public static double[] transform(double[] oldValues){
        
        double[] newValues = new double[oldValues.length];
        
        // Could just use sort, but this is more robust in that it handles infinity, nan, and 
        // has explicit policy for ties.  Also, this policy matches what the Stuart lab R 
        // script does thus ensuring this code produces same output. 
        NaturalRanking ranking = new NaturalRanking(NaNStrategy.FIXED,TiesStrategy.MAXIMUM);    
        //  input: (20, 17, 30, 42.3, 17, 50, Double.NaN, Double.NEGATIVE_INFINITY, 17)
        //  output: (6, 5, 7, 8, 5, 9, 2, 2, 5)
        double[] ranks = ranking.rank(oldValues); 
        
        // count non-nan values...
        double numValues = ranks.length - countNAN(oldValues);
        double invsum = (double)(1.0/numValues);
        
        for (int i = 0; i < oldValues.length; i++){
            double rank = ranks[i];
            double scaled = rank*invsum - invsum;
            double newVal = Math.abs(exp.inverseCumulativeProbability(Math.abs(scaled)));
            newValues[i] = newVal;
        }
        return(newValues);
    }
    
    static double countNAN(double[] vals){
        int count = 0;
        for(int i = 0;i < vals.length;i++){
            double val = vals[i];
            if (val == Double.NaN) count++;
        }
        return(count);
    }
}
