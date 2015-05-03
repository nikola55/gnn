#ifndef TRAININGSET_H_
#define TRAININGSET_H_

#include <gsl/gsl_matrix.h>

namespace gnn {

class TrainingSet {

public:

	TrainingSet();

	virtual ~TrainingSet();

	const gsl_matrix * getTrainingExamples() const;

	const gsl_matrix * getClasses() const;

	void setTrainingExamples(gsl_matrix * a);

	void setClasses(gsl_matrix * y);

	int getTrainingExamplesCount() const;

	int getFeaturesCount() const;

	int getClassesCount() const;

	static TrainingSet * loadFromFile(
								const char *dataFile,
								int examplesCount,
								int featuresCount,
								const char *lebelsFile,
								int classesCount);

private:
	
	static void loadClasses(gsl_matrix_int *yLebels, gsl_matrix * classes);
	
	static void loadClassesBinary(gsl_matrix_int *yLebels, gsl_matrix * classes);
	
	gsl_matrix * trainingExamples;
	gsl_matrix * classes;

};

} /* namespace gnn */

#endif /* TRAININGSET_H_ */
