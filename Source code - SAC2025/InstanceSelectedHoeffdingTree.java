package moa.classifiers.trees;

import java.util.LinkedList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;

/*
 * Explanation of the instance selection method:
 * If the algorithm classifies the instance correctly, the method adopts a probabilistic approach, similar to flipping a coin, 
 * to decide whether this instance should be included in the training. 
 * If the randomly generated value between 0 and 1 is higher than the established control factor (parameter a - ControlFactorIS), the instance is then used for training. 
 * Thus, the instance Xi is selected for training if the following condition is satisfied: r > a, where a controls the rigor of the selection. 
 * */
public class InstanceSelectedHoeffdingTree extends HoeffdingTree{

	@Override
	public String getPurposeString() {
		return "Regularized EFDT for evolving data streams from xxxxxxxxxxxxxxxxxxxx.";
	}
	
	public FloatOption controlFactorIS = new FloatOption(
			"ControlFactorIS",
			'i',
			"Control factor for instance selection (1 = Correctly classified instances are not used for training)",
			0.0000001, 0.0, 1.0);
	
	public IntOption seedTrainingInstance  = new IntOption("seedTrainingInstance", 'a',
            "Random seed used to select a training instance.", 1, 0, Integer.MAX_VALUE);
	
	private static final long serialVersionUID = 1L;

    protected long InstancesForTraining;
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (this.treeRoot == null) {
    		this.classifierRandom.setSeed(this.seedTrainingInstance.getValue());
    		this.InstancesForTraining = 0;
    	}
    	
    	DoubleVector vote = new DoubleVector(getVotesForInstance(inst));

    	int trueClass = (int) inst.classValue();
    	int predictedClass = Utils.maxIndex(vote.getArrayRef());

    	boolean usedToTrain = true;
    	double value = this.controlFactorIS.getValue();
    	if (value == 1) { //use only misclassified 
    		usedToTrain = (trueClass!=predictedClass);
    	}
    	else if (trueClass == predictedClass) {
    		double	random = this.classifierRandom.nextDouble();
	    	usedToTrain = (random>value);
    	}
    	
    	if (usedToTrain) {
    		this.InstancesForTraining++;	 
    		super.trainOnInstanceImpl(inst);
    	}
    }
    
    
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
    	
    	List<Measurement> measurementList = new LinkedList<Measurement>();
        Measurement[] measurementSuper = super.getModelMeasurementsImpl();    
    	for (Measurement measurement : measurementSuper) {
    		measurementList.add(measurement);
        }
    	measurementList.add(new Measurement("instances used for training", this.InstancesForTraining));
    	return measurementList.toArray(new Measurement[measurementList.size()]);
    	
    }
    
    @Override
    public boolean isRandomizable() {
        return true;
    }
	 
}
