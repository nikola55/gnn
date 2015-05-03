#ifndef PERSISTER_
#define PERSISTER_

namespace gnn {

class NeuralNet;
class Def;

class IOHandler {

public:

	static void store(NeuralNet *net, Def * def, const char *fileName);
	
	static NeuralNet * load(const char *fileName);

};


} /* namespace gnn */
#endif