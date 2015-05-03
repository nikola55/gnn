/*
 * MemHandler.cpp
 *
 *  Created on: 18.03.2015 ã.
 *      Author: nikola.petkanski
 */

#include <gsl/gsl_matrix.h>

#include "MemHandler.h"
#include "Def.h"

namespace gnn {

MemHandler::MemHandler(Def *def)
{
	this->def = def;
}

gsl_matrix ** MemHandler::allocNeuronSlots(int exampleCount, bool addBias)
{
	gsl_matrix **neuronSlots = new gsl_matrix*[def->getLayersCount() - 1];

	int biasTerm = addBias ? 1 : 0;

	for (int i = 1; i < def->getLayersCount(); ++i)
	{
		neuronSlots[i-1] = gsl_matrix_alloc(exampleCount, def->getUnitCount(i+1) + biasTerm);
	}

	return neuronSlots;
}

void MemHandler::freeNeuronSlots(gsl_matrix **neuronSlots)
{
	for (int i = 0; i < def->getLayersCount() - 1; ++i)
	{
		gsl_matrix_free(neuronSlots[i]);
	}

	delete []neuronSlots;
}

gsl_matrix ** MemHandler::allocWeights()
{
	gsl_matrix **Weights = new gsl_matrix*[def->getLayersCount() - 1];

	for (int i = 0; i < def->getLayersCount() - 1; ++i)
	{

		if(i == 0) // The training set values must be passed with bias term
			Weights[i]=gsl_matrix_alloc(def->getUnitCount(i+1), def->getUnitCount(i+2));
		else
			Weights[i]=gsl_matrix_alloc(def->getUnitCount(i+1) + 1, def->getUnitCount(i+2));

	}

	return Weights;
}

void MemHandler::freeWeights(gsl_matrix ** Weights)
{
	for (int i = 0; i < def->getLayersCount() - 1; ++i)
	{
		gsl_matrix_free(Weights[i]);
	}
	delete []Weights;
}

MemHandler::~MemHandler()
{


}

} /* namespace gnn */
