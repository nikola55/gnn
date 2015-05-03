#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <gsl/gsl_matrix.h>

namespace gnn {

class Def;
class TrainingSet;
class MemHandler;

class NeuralNet {

	friend class IOHandler;

public:

	NeuralNet(Def *def, TrainingSet *ts);
	
	NeuralNet(Def * def);

	virtual ~NeuralNet();

	void feedForward(gsl_matrix **Weights);

	void backpropagate();

	double computeCost();

	gsl_matrix ** getGradients();

	gsl_matrix ** getWeights();

	gsl_matrix * getOutput();

	void predict(const gsl_matrix * input, gsl_matrix ** Weights);
	
	Def * getDef();

private:

	void sigmoid(gsl_matrix *Z, gsl_matrix *A, gsl_matrix *ABias);

	void forwardPropagate(
			gsl_matrix **Weights, const gsl_matrix *input,
			gsl_matrix **ZOuts, gsl_matrix **AOuts, gsl_matrix **AOutsBias);

	void randomizeWeights();

	void randomizeMatrix(gsl_matrix *mat);
	
	void nullifyPtr();

	void allocZOuts();

	void allocAOuts();

	void allocGradients();

	void allocDeltas();

	void allocWeights();

	void freeZOuts();

	void freeAOuts();

	void freeGradients();

	void freeDeltas();

	void freeWeights();

	void computeDelta(int layer);

	void computeDeltas();

private:

	Def * def;
	TrainingSet * ts;

	MemHandler * memHandler;

	gsl_matrix ** ZOuts;
	gsl_matrix ** AOuts;
	gsl_matrix ** AOutsBias;

	gsl_matrix ** Weights;
	gsl_matrix ** Gradients;

	gsl_matrix ** Deltas;

	bool initialOutput;
	gsl_matrix **currentOutput;

};

} /* namespace gnn */

#endif /* NEURALNET_H_ */
