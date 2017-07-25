#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

struct Connection 
{
	double weight;
	double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned numOuputs, unsigned myIndex);
	void feedForward(const Layer &prevLayer);
	
	void setOutputValue(double input) { outputValue = input; }
	double getOutputValue() const { return outputValue; }
	
	 double randomWeight(void)
	{
		 return rand()/(double)RAND_MAX;
	}

private:
	static double transferFunction(double x)
	{
		return tanh(x);
	}
	static double transferFunctionDerivative(double x)
	{
		return (1 - x*x);
	}
	double outputValue;
	vector<Connection> inputConnections;
	unsigned myIndex;
};


class Net
{
public:
	Net(const vector<unsigned> &topology);	//constructor to initialize net with given topology
	void feedForward(const vector<double> &inputValues);
	void backprop(const vector<double> &targetValues);
	void getResults(vector<double> &results) const;

private:
	vector<Layer> layers; //layers_[layerNum][neuronNum]

};

