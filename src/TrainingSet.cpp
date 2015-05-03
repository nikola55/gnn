#include "TrainingSet.h"

namespace gnn {

TrainingSet::TrainingSet()
{
	trainingExamples = NULL;
	classes = NULL;
}

const gsl_matrix * TrainingSet::getClasses() const
{
	return classes;
}

const gsl_matrix * TrainingSet::getTrainingExamples() const
{
	return trainingExamples;
}

void TrainingSet::setClasses(gsl_matrix *y)
{
	classes = y;
}

void TrainingSet::setTrainingExamples(gsl_matrix *a)
{
	trainingExamples = a;
}

int TrainingSet::getClassesCount() const
{
	return classes->size2;
}

int TrainingSet::getTrainingExamplesCount() const
{
	return trainingExamples->size1;
}

int TrainingSet::getFeaturesCount() const
{
	return trainingExamples->size2;
}

void TrainingSet::loadClasses(gsl_matrix_int *yLebels, gsl_matrix * classes)
{
	for (unsigned int i = 0; i < yLebels->size1; ++i)
	{
		gsl_matrix_set(classes, i, gsl_matrix_int_get(yLebels, i, 0), 1.0);
	}
}

void TrainingSet::loadClassesBinary(gsl_matrix_int *yLebels, gsl_matrix * classes)
{
	for (unsigned int i = 0; i < yLebels->size1; ++i)
	{
		gsl_matrix_set(classes, i, 0, gsl_matrix_int_get(yLebels, i, 0));
	}
}

TrainingSet * TrainingSet::loadFromFile(
							const char *dataFile,
							int examplesCount,
							int featuresCount,
							const char *lebelsFile,
							int classesCount)
{

	TrainingSet *ts = new TrainingSet();

	gsl_matrix *xData = gsl_matrix_alloc(examplesCount, featuresCount);

	FILE * inputData = fopen(dataFile, "rb");

	if(!inputData)
	{

	}

	gsl_matrix_fread(inputData, xData);

	fclose(inputData);

	gsl_matrix_int *yLebels = gsl_matrix_int_alloc(examplesCount, 1);
	FILE * inputLebels = fopen(lebelsFile, "rb");

	if(!inputLebels)
	{

	}

	gsl_matrix_int_fread(inputLebels, yLebels);
	fclose(inputLebels);

	gsl_matrix *Y;
	
	if(0)
	{
		Y = gsl_matrix_calloc(examplesCount, 1);
		loadClassesBinary(yLebels, Y);
	}
	else 
	{
		Y = gsl_matrix_calloc(examplesCount, classesCount);
		loadClasses(yLebels, Y);
	}
	

	gsl_matrix_int_free(yLebels);

	ts->setTrainingExamples(xData);
	ts->setClasses(Y);

	return ts;

}

TrainingSet::~TrainingSet()
{
	gsl_matrix_free(trainingExamples);
	gsl_matrix_free(classes);
}

} /* namespace gnn */
