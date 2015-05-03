#include "Def.h"

#include <iostream>

namespace gnn {

Def::Def(int inputLayerUnitCount, int outputLayerUnitCount) :
		inputLayerUnitCount(inputLayerUnitCount),
		outputLayerUnitCount(outputLayerUnitCount),
		nLayers(3)
{

	this->unitCount = new int[nLayers];

	unitCount[0] = inputLayerUnitCount;
	unitCount[1] = (inputLayerUnitCount - outputLayerUnitCount) / 2;
	unitCount[2] = outputLayerUnitCount;

}

Def::Def(int inputLayerUnitCount, int nLayers, int outputLayerUnitCount) :
		inputLayerUnitCount(inputLayerUnitCount),
		outputLayerUnitCount(outputLayerUnitCount),
		nLayers(nLayers)
{

	this->unitCount = new int[nLayers];

	unitCount[0] = inputLayerUnitCount;
	unitCount[nLayers - 1] = outputLayerUnitCount;

	for (int i = 1; i < nLayers-1; ++i) {
		unitCount[i] = -1;
	}

}

int Def::getUnitCount(int layer) const
{
	if(layer <= 0 || layer > nLayers) {
		std::cerr << "Invalid layer: " << layer;
	}

	return this->unitCount[layer-1];
}

void Def::setUnitCount(int layer, int count)
{
	if(layer <= 1 || layer >= nLayers) {
		std::cerr << "Invalid layer: " << layer;
	}

	this->unitCount[layer-1] = count;
}

int Def::getLayersCount() const {

	return nLayers;

}

bool Def::isValid()
{

	for (int i = 0; i < nLayers; ++i) {
		if(unitCount[i] <= 0) return false;
	}

	return true;
}

Def::~Def()
{
	delete []unitCount;
}

} /* namespace gnn */
