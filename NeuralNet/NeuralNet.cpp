// NeuralNet.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "NeuralNet.h"
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

//*************************	 NEURON	*****************************************
Neuron::Neuron(unsigned numOuputs, unsigned Index)
{
	for (int c = 0; c < numOuputs; ++c)
	{
		inputConnections.push_back(Connection());
		inputConnections.back().weight = randomWeight();
		cout << "weight" << inputConnections.back().weight;
	}

	myIndex = Index;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;
	for (unsigned i = 0; i < prevLayer.size(); ++i)
	{
		sum += prevLayer[i].getOutputValue() * prevLayer[i].inputConnections[myIndex].weight;
	}

	outputValue = Neuron::transferFunction(sum);
	//cout << "op here --> " << sum << endl;
}


//***************************************	NET	 ****************************************************
Net::Net(const vector<unsigned> &topology)
{
	unsigned totalLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < totalLayers; ++layerNum)
	{
		layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a neuron at layer " << layerNum <<endl;
		}
	}
}

void Net::feedForward(const vector<double> &inputValues)
{
	assert(inputValues.size() == layers[0].size() - 1);

	//assign inital layer's outputs
	for (unsigned i = 0; i < inputValues.size(); ++i)
	{
		layers[0][i].setOutputValue(inputValues[i]);
	}

	//forward prop
	for (unsigned layerNum = 1; layerNum < layers.size(); ++layerNum)
	{
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n)
		{
			layers[layerNum][n].feedForward(prevLayer);
		}
	}

}

void Net::backprop(const vector<double> &targetValues)
{
	//get root mean square error
	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		error += delta*delta;
	}

	error = error / (outputLayer.size() - 1);
	error = sqrt(error);
}


int main()
{
	vector<double> inputValues;
	vector<unsigned> topology;

	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net myNet(topology);
	
	inputValues.push_back(0.3);
	inputValues.push_back(0.8);
	inputValues.push_back(0.1);

	myNet.feedForward(inputValues);

	int q;
	cin >> q;
    return 0;
}

