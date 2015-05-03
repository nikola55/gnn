#include <math.h>
#include <time.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>


#include "TrainingSet.h"
#include "Def.h"
#include "MemHandler.h"
#include "NeuralNet.h"

namespace gnn {

NeuralNet::NeuralNet(Def *def, TrainingSet *ts) 
	: def(def), ts(ts)
{
	memHandler = new MemHandler(def);
	initialOutput = false;

	nullifyPtr();

}

NeuralNet::NeuralNet(Def * def)
	: def(def), memHandler(new MemHandler(def))
{

	initialOutput = false;
	nullifyPtr();
	
}

Def * NeuralNet::getDef()
{
	return def;
}

void NeuralNet::nullifyPtr()
{
	ZOuts = NULL;
	AOuts = NULL;
	AOutsBias = NULL;

	Weights = NULL;
	Gradients = NULL;
	Deltas = NULL;

	currentOutput = NULL;
}

double NeuralNet::computeCost()
{

	double cost = 0.0;

	double m = ts->getTrainingExamplesCount();

	gsl_matrix *Outputs = AOuts[def->getLayersCount() - 2];

	for (unsigned int i = 0; i < Outputs->size1; ++i) {
		for (unsigned int j = 0; j < Outputs->size2; ++j) {

			double a = gsl_matrix_get(Outputs, i, j);
			double y = gsl_matrix_get(ts->getClasses(), i, j);

			cost += -y * log(a) - (1.0 - y) * log(1.0 - a);
		}
	}

	return 1.0 / m * cost;

}

void NeuralNet::sigmoid(gsl_matrix *Z, gsl_matrix *A, gsl_matrix *ABias)
{

	for (unsigned int i = 0; i < Z->size1; ++i)
	{

		gsl_matrix_set(ABias, i, 0, 1.0);

		for (unsigned int j = 0; j < Z->size2; ++j)
		{
			double z = gsl_matrix_get(Z, i, j);
			double a = 1.0 /  (1.0 + exp(-z));

			gsl_matrix_set(A, i, j, a);
			gsl_matrix_set(ABias, i, j+1, a);
		}
	}
}

gsl_matrix ** NeuralNet::getWeights()
{
	if(Weights == NULL)
	{
		allocWeights();
		randomizeWeights();
	}

	return Weights;
}

void NeuralNet::randomizeMatrix(gsl_matrix *mat)
{

	gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set (r, time(NULL));

	for(unsigned int i = 0 ; i < mat->size1 ; i++)
	{
		for(unsigned int j =0 ; j < mat->size2 ; j++)
		{
			double rand = gsl_rng_uniform (r) - 0.5;

			gsl_matrix_set(mat, i, j, rand);

		}
	}

	gsl_rng_free (r);
}

gsl_matrix * NeuralNet::getOutput()
{
	return currentOutput[def->getLayersCount() - 2];
}

void NeuralNet::randomizeWeights()
{
	for (int i = 0; i < def->getLayersCount() - 1; ++i)
	{
		randomizeMatrix(Weights[i]);
	}
}

void NeuralNet::feedForward(gsl_matrix **Weights)
{
	
	if(ZOuts == NULL && AOuts == NULL)
	{
		allocZOuts();
		allocAOuts();
	}
	
	forwardPropagate(Weights, ts->getTrainingExamples(),
			ZOuts, AOuts, AOutsBias);

}

void NeuralNet::forwardPropagate(
		gsl_matrix **Weights, const gsl_matrix *input,
		gsl_matrix **ZOuts, gsl_matrix **AOuts, gsl_matrix **AOutsBias)
{

	const gsl_matrix *A = input;

	for (int i = 0; i < def->getLayersCount() - 1; ++i)
	{
		gsl_matrix *Z = ZOuts[i];
		gsl_matrix *W = Weights[i];

		gsl_blas_dgemm ( CblasNoTrans, CblasNoTrans, 1.0, A, W, 0.0, Z);

		sigmoid(Z, AOuts[i], AOutsBias[i]);

		A = AOutsBias[i];

	}

	currentOutput = AOuts;
	initialOutput = true;
}

void NeuralNet::predict(const gsl_matrix *input, gsl_matrix ** Weights)
{
	gsl_matrix **outputZ = memHandler->allocNeuronSlots(input->size1, false);
	gsl_matrix **outputA = memHandler->allocNeuronSlots(input->size1, false);
	gsl_matrix **outputABias = memHandler->allocNeuronSlots(input->size1, true);

	forwardPropagate(Weights, input, outputZ, outputA, outputABias);

	currentOutput = outputA;
	initialOutput = false;

	memHandler->freeNeuronSlots(outputZ);
	memHandler->freeNeuronSlots(outputABias);
}

void NeuralNet::computeDelta(int layer)
{

	gsl_matrix *Delta_large = gsl_matrix_alloc(ts->getTrainingExamplesCount(), def->getUnitCount(layer) + 1);

	gsl_matrix *Delta_next = Deltas[layer - 1];
	gsl_matrix *Weight_current = Weights[layer - 1];

	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, Delta_next, Weight_current, 0.0, Delta_large);

	gsl_matrix *Delta_current = Deltas[layer - 2];

	const gsl_matrix *A = AOuts[layer - 2];

	for (unsigned int i = 0; i < Delta_current->size1; ++i)
	{
		for (unsigned int j = 0; j < Delta_current->size2; ++j)
		{
			double a = gsl_matrix_get(A, i, j);
			double d = gsl_matrix_get(Delta_large, i, j+1);

			double s = d * a * ( 1.0 - a );

			gsl_matrix_set(Delta_current, i, j, s);
		}
	}

	gsl_matrix_free(Delta_large);

}

void NeuralNet::computeDeltas() {

	gsl_matrix* Delta = Deltas[def->getLayersCount() - 2];
	gsl_matrix* A = AOuts[def->getLayersCount() - 2];
	gsl_matrix_memcpy(Delta, A);
	gsl_matrix_sub(Delta, ts->getClasses());

	for (int i = def->getLayersCount() - 1; i >= 2; i--) {
		computeDelta(i);
	}
}

void NeuralNet::backpropagate()
{

	if(Deltas == NULL)
	{
		allocDeltas();
	}

	computeDeltas();

	if(Gradients == NULL)
	{
		allocGradients();
	}

	double m = ts->getTrainingExamplesCount();

	for (int i = 0; i < def->getLayersCount() - 1; ++i) {

		const gsl_matrix * A = i == 0 ? ts->getTrainingExamples() : AOutsBias[i-1];

		gsl_matrix * Delta = Deltas[i];
		gsl_matrix *Gradient = Gradients[i];

		gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0 / m, A, Delta, 0.0, Gradient);

	}

}

gsl_matrix ** NeuralNet::getGradients()
{
	return Gradients;
}

void NeuralNet::allocWeights()
{

	Weights = memHandler->allocWeights();
}

void NeuralNet::freeWeights()
{
	memHandler->freeWeights(Weights);
}


void NeuralNet::allocGradients()
{
	Gradients = memHandler->allocWeights();
}

void NeuralNet::freeGradients()
{
	memHandler->freeWeights(Gradients);
}

void NeuralNet::allocDeltas()
{

	Deltas = memHandler->allocNeuronSlots(ts->getTrainingExamplesCount(), false);
}

void NeuralNet::freeDeltas()
{
	memHandler->freeNeuronSlots(Deltas);
}

void NeuralNet::allocZOuts()
{
	ZOuts = memHandler->allocNeuronSlots(ts->getTrainingExamplesCount(), false);
}

void NeuralNet::freeZOuts()
{
	memHandler->freeNeuronSlots(ZOuts);
}

void NeuralNet::allocAOuts()
{
	AOuts = memHandler->allocNeuronSlots(ts->getTrainingExamplesCount(), false);
	AOutsBias = memHandler->allocNeuronSlots(ts->getTrainingExamplesCount(), true);
}

void NeuralNet::freeAOuts()
{
	memHandler->freeNeuronSlots(AOuts);
	memHandler->freeNeuronSlots(AOutsBias);
}

NeuralNet::~NeuralNet()
{

	if(ZOuts != NULL)
		freeZOuts();

	if(AOuts != NULL)
		freeAOuts();

	if(Weights != NULL)
		freeWeights();

	if(Deltas != NULL)
		freeDeltas();

	if(Gradients != NULL)
		freeGradients();

	if(!initialOutput && currentOutput != NULL)
		memHandler->freeNeuronSlots(currentOutput);

	delete memHandler;

}

} /* namespace gnn */
