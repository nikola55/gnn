#include "neural_net.h"
#include "matrix_opers.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
/*
 * Allocate enough space to hold each output neuron
 * for each layer but the input
 *
 * @param1 definition
 * @param2 number of rows
 * @param3 add space for bias term
 *
 */

double **allocNeuronSlots(gnn::nnDef def, int nRows, bool addCol)
{
	double **ns = new double*[def.nLayers-1];
	for(int i = 1 ; i < def.nLayers ; i++) {
		int nCols = def.nodes(i) + addCol;
		ns[i-1] = new double[nRows*nCols];
	}
	return ns;
}

void freeNeuronSlots(gnn::nnDef def, double **ns)
{
	for(int i = 0 ; i < def.nLayers - 1 ; i++) {
		delete []ns[i];
	}
	delete []ns;
}

/*
 * @param1 weighted sum of the neurons
 * @param2 storage for the activations
 * @param3 storage for the activations with bias term added
 * @param4 dimension 1 of Z
 * @param5 dimension 2 of Z
 * calculates the the activations for each neuron using:
 * a = 1 / ( 1 + exp(-z) ). Produces 2 matrices
 * param2 the activations
 * param3 with bias (1) term added
 */

void computeActivations(const double *Z,
						double *Activations,
						double *Ab,
						int ZNRows, int ZNCols
						)
{
	int nElems = ZNRows*ZNCols;
	int AbNCols = ZNCols+1;
	for(int i = 0 ; i < ZNRows ; i++) {
		int AbRow = i*AbNCols;
		Ab[AbRow] = 1;
		for(int j = 0 ; j < ZNCols ; j++) {
			int elem = i*ZNCols + j;
			double a = 1.0f / (1.0f + exp(-Z[elem]));
			Activations[elem] = a;
			Ab[AbRow+j+1] = a;
		}
	}
}

/*
 * @param1 definition
 * @param2 input layer
 * @param3 number of examples in input layer
 * @param4 weights associated with each layer ( n(layers) - 1 )
 * @param5 stroage for the activations for each of the layers ( n(layers) - 1 )
 * 		   need to be preallocated
 * @param6 storage for the activation with bias added +1 column
 *
 * Performs forward propagation on the given training set (param2)
 * computes the activation of all layers and stores the result in
 * param param5
 */

void gnn::feedForward(const nnDef &def,
					  const double *inputLayer,
					  const double **Weigths,
					  int inputLayerRows,
					  double **Activations,
					  double **ActivationsBias
					  )
{
	// Allocate enough space for n(layers)-1
	double **Z = allocNeuronSlots(def, inputLayerRows, false);

	const double * Acurrent = inputLayer;
	int aCurrentNRows = inputLayerRows;

	for(int i = 0 ; i < def.nLayers - 1 ; i++) {

		int aCurrentNCols = def.nodes(i) + 1;
		int zNCols = def.nodes(i+1);

		mmMultiply(aCurrentNRows, aCurrentNCols,
				   Acurrent, false,
				   aCurrentNCols, zNCols,
				   Weigths[i], false,
				   Z[i], 1.0);

		computeActivations(Z[i], Activations[i], ActivationsBias[i],
						   inputLayerRows, zNCols
                            );

		Acurrent = ActivationsBias[i];
	}

	freeNeuronSlots(def, Z);
}

/*
 * @param1 definition
 * @param2 output layer activations
 * @param3 the class associated with each training example
 *		   (! in the same layout as param2)
 * @param4 number of param2 in classes and param3
 *
 * @return the cost
 *
 * computes the weighted sum of the difference between the predicted
 * values (param2) and the actual values (param3) using the function
 */

double gnn::cost(const gnn::nnDef &def,
				const double *outputLayers,
				const double *classes,
				int outputLayerRows
				)
{

	int clCount = def.nodes(def.nLayers - 1);

	double cost = 0.0f;
	for(int i = 0 ; i < outputLayerRows ; i++) {
		for(int j = 0 ; j < clCount ; j++) {
			int elem = i*clCount + j;
			cost +=
				-classes[elem] * log(outputLayers[elem])
				-(1.0 - classes[elem]) * log(1.0 - outputLayers[elem]);
		}
	}
	return cost/outputLayerRows;
}

/*
 * Computes the error associated with layer param2
 *
 * if param2 != param1.nLayers-1
 *
 * delta = ( weight^T * delta_next ) .* sigmoid'
 *
 * @param1 definition
 * @param2 index of the layer
 * @param3 Weigths of layer param2
 * @param4 Error of the layer param2 + 1
 * @param5 activations of layer param2
 * @param6 storage for the result
 * @param7 number of rows
 *
 * if param2 < param1.nLayers-1
 *
 * delta = a_output_layer - classes
 *
 * @param1 definition
 * @param2 index of the layer
 * @param3 unused
 * @param4 the classes associate with each example
 * @param5 activations of the output layer
 * @param6 storage for the result
 * @param7 number of rows
 */

void computeDelta(const gnn::nnDef &def,
				  int layer,
				  const double *weight,
				  const double *deltaNext,
				  const double *activations,
				  double *delta,
				  int numOfRows
				  )
{

	if(0 > layer || layer >= def.nLayers) {
		fprintf(stderr, "computeDelta: Invalid param2: %d\n", layer);
		exit(1);
	}

	int deltaNRows = numOfRows;
	int deltaNCols = def.nodes(layer);

	if(layer == def.nLayers-1) {
		gnn::mmSubtract(deltaNRows, deltaNCols, activations,
						deltaNRows, deltaNCols, deltaNext,
						delta );
		return;
	}

	double * work = new double[deltaNRows*(deltaNCols+1)];

	int deltaNextNRows = numOfRows;
	int deltaNextNCols = def.nodes(layer+1);

	int weightsNRows = def.nodes(layer);
	int weightsNCols = def.nodes(layer+1);

	gnn::mmMultiply(deltaNextNRows, deltaNextNCols,
				    deltaNext, false,
				    weightsNRows, weightsNCols,
				    weight, true,
				    work, 1.0 );

	for(int i = 0 ; i < deltaNRows ; i++) {
		int row = i*deltaNCols;
		int workRow = i*deltaNCols+1;
		for(int j = 0 ; j < deltaNCols ; j++) {
			int elem = row+j;
			double val = work[workRow+j+1];
			register double a = activations[elem];
			delta[elem] = val * a * ( 1 - a );
		}
	}

	delete []work;
}

/*
* computes the gradient associated with each weight ( param4 ) and stores the
* result in param4 using backpropagation
*
* @param1 definition
* @param2 computed activations for each layer including the training set ( n(layers) )
* @param3 activations with bias term added
* @param4 tha classes for each example in the training set
* @param5 weights associated with each layer ( n(layer) - 1 )
* @param6 gradients of each of the weights
* @param7 number of rows in the input
*/

void gnn::backpropagate(const gnn::nnDef &def,
					    const double **Activations,
					    const double **ActivationsBias,
					    const double *classes,
					    const double **Weigths,
					    double **Gradients,
					    int inputLayerRows
					    )
{

	int currentLayer = def.nLayers-1;
	double *pDelta = new double[inputLayerRows*def.nodes(currentLayer)];
	computeDelta(def, currentLayer, 0,
				 classes, Activations[currentLayer],
				 pDelta, inputLayerRows );

	 for(; currentLayer >= 1 ; currentLayer--) {

		int cGrad = currentLayer-1;
		double *Grad = Gradients[cGrad];
		const double *cActB = ActivationsBias[cGrad];
		int cActBNCols = def.nodes(cGrad)+1;

		mmMultiply(inputLayerRows,
				   def.nodes(cGrad)+1,
				   cActB, true, inputLayerRows,
				   def.nodes(currentLayer),
				   pDelta, false, Grad,
				   1.0f / inputLayerRows );

		if(currentLayer == 1) continue;

		double *nDelta = new double[inputLayerRows*def.nodes(cGrad)];

		computeDelta(def, cGrad, Weigths[cGrad],
					 pDelta, Activations[cGrad],
					 nDelta, inputLayerRows );

		delete []pDelta;
		pDelta = nDelta;
	 }

	 delete []pDelta;
}

gnn::nnDef::nnDef(int nlay, const int* layn) :
	nl(nlay), nLayers(nl), lc(new int[nl])
{
	copy(layn);
}

gnn::nnDef::nnDef(const nnDef &def) :
	nl(def.nl), nLayers(nl), lc(new int[nl])
{
	copy(def.lc);
}

gnn::nnDef& gnn::nnDef::operator=(const nnDef &def)
{
	delete []lc;
	nl = def.nl;
	lc = new int[nl];
	copy(def.lc);
	return *this;
}

gnn::nnDef::~nnDef()
{
	delete []lc;
}

void gnn::nnDef::copy(const int *layerNodes)
{
	for(int i = 0 ; i < nLayers ; i++) {
		lc[i]=layerNodes[i];
	}
}

int gnn::nnDef::nodes(int l) const
{
	if(0 > l || l >= nLayers) {
		return -1;
	}
	return lc[l];
}
