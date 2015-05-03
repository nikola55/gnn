#include "IOHandler.h"

#include "Def.h"
#include "TrainingSet.h"
#include "NeuralNet.h"
#include <gsl/gsl_matrix.h>

namespace gnn {

/**
 *
 *  Store in the following format
 *  0   4   8      4+4n
 *  +---+---+---+---+---+---+---+
 *  | n |l1c|...|lnc| w1| ..|wn-1|
 *  +---+---+---+---+---+---+---+
 *	Where :
 *  n - number of layers
 *  lm - layer m units count
 *  wm - weight for layer m
 *
 */

void IOHandler::store(NeuralNet *net, Def * def, const char *fileName)
{

	FILE * f = fopen(fileName, "wb");
	
	int nUnits = def->getLayersCount();
	
	fwrite(&nUnits, sizeof(int), 1, f);
	
	for(int i = 1 ; i <= nUnits ; i++)
	{
		int cLayerUnits = def->getUnitCount(i);
		fwrite(&cLayerUnits, sizeof(int), 1, f);
		
	}
	
	gsl_matrix ** Weigths = net->getWeights();
	
	for(int i = 0 ; i < nUnits - 1 ; i++)
	{
		gsl_matrix_fwrite(f, Weigths[i]);
	}
	
	fclose(f);
	
}

Def * readDef(FILE * f)
{
	
	int nLayers = 0;
	fread(&nLayers, sizeof(int), 1, f); // read the number of units
	int * layersUnits = new int[nLayers];

	for(int i = 0 ; i < nLayers; i++)
	{
		fread(&layersUnits[i], sizeof(int), 1, f);
	}
	
	Def * def = new Def(layersUnits[0], nLayers, layersUnits[nLayers-1]);
	
	for (int i = 2; i < def->getLayersCount(); ++i)
	{
		def->setUnitCount(i, layersUnits[i-1]);
	}

	delete []layersUnits;

	return def;
}

NeuralNet * IOHandler::load(const char *fileName)
{
	
	FILE * f = fopen(fileName, "rb");

	Def * def = readDef(f);
	
	NeuralNet * net = new NeuralNet(def);
	
	gsl_matrix ** Weigths = net->getWeights();
	
	int nLayers = def->getLayersCount();
	
	for(int i  = 0 ; i < nLayers - 1 ; i++)
	{
		gsl_matrix_fread(f, Weigths[i]);
	}
	
	fclose(f);
	
	return net;
}



}
