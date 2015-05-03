#ifndef DEF_H_
#define DEF_H_

#include <gsl/gsl_matrix.h>

namespace gnn {

class Def {

public:

	Def(int inputLayerUnitCount, int outputLayerUnitCount);

	Def(int inputLayerUnitCount, int nLayers, int outputLayerUnitCount);

	void setUnitCount(int layer, int count);

	int getUnitCount(int layer) const;

	int getLayersCount() const;

	bool isValid();

	virtual ~Def();

private:

	int inputLayerUnitCount;
	int outputLayerUnitCount;

	int nLayers;

	int * unitCount;

};

} /* namespace gnn */

#endif /* DEF_H_ */
