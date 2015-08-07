#ifndef _GNN_H_INCLUDED_
#define _GNN_H_INCLUDED_
	
namespace gnn {
	
	/*
	 * The definition of the neural network.
	 */ 
	class nnDef {
		int nl;
		int *lc;
		void copy(const int *layerNodes);
	public:
		/*
		*  @param1 number of layers including the input and output layer
		*  @param2 int array with length param1 - number of nodes for each layer 
		*/
		nnDef(int nlay, const int *layn);
		nnDef(const nnDef &def);
		nnDef & operator=(const nnDef &def);
		~nnDef();

		int nodes(int) const ;
		const int &nLayers;
	};
	
	/*
	* Performs forward propagation on the given training set (param2)
	* computes the activation of all layers and stores the result in
	* param5.
	*
	* @param1 definition
	* @param2 input layer 
	* @param3 number of examples in input layer
	* @param4 weights associated with each layer ( n(layers) - 1 )
	* @param5 stroage for the activations for each of the layers ( n(layers) - 1 )
	* 		  need to be preallocated. 
	* @param6 storage for the activation with bias added +1 column
	*/
	 
	void feedForward(const nnDef &def,
					 const float *inputLayer,
					 const float **Weigths,
					 int inputLayerRows,
					 float **Activations,
					 float **ActivationsBias
					 );
	
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
	
	float cost(const gnn::nnDef &def,
			   const float *outputLayers,
			   const float *classes,
			   int outputLayerRows
			   );
	
	
	/*
	* computes the gradient associated with each weight ( param4 ) and stores the
	* result in param4 using backpropagation 
	*
	* @param1 definition
	* @param2 computed activations for each layer including the training set ( n(layers) ) { &ts[0], &A1[0], ... &AN[0] }
	* @param3 activations with bias term added including the training set ( n(layers) ) { &ts[0], &A1[0], ... &AN[0] }
	* @param4 tha classes for each example in the training set
	* @param5 weights associated with each layer ( n(layer) - 1 )
	* @param6 gradients of each of the weights
	* @param7 number of rows in the input
	*/
	
	void backpropagate(const gnn::nnDef &def,
					   const float **Activations,
					   const float **ActivationsBias,
					   const float *classes,
					   const float **Weigths,
					   float **Gradients,
					   int inputLayerRows
					   );
	
}
	
#endif