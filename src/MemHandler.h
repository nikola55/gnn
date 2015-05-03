/*
 * MemHandler.h
 *
 *  Created on: 18.03.2015 ã.
 *      Author: nikola.petkanski
 */

#ifndef MEMHANDLER_H_
#define MEMHANDLER_H_

#include <gsl/gsl_matrix.h>

namespace gnn {

class Def;

class MemHandler {
public:

	MemHandler(Def * def);

	gsl_matrix ** allocNeuronSlots(int exampleCount, bool addBias);

	void freeNeuronSlots(gsl_matrix **neuronSlots);

	gsl_matrix ** allocWeights();

	void freeWeights(gsl_matrix ** Weights);

	virtual ~MemHandler();


private:

	Def * def;

};

} /* namespace gnn */

#endif /* MEMHANDLER_H_ */
