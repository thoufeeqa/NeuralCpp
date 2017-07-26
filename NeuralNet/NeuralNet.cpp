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

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += inputConnections[n].weight * nextLayer[n].gradient;
	}

	return sum;
		
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - outputValue;
	gradient = delta * transferFunctionDerivative(outputValue);
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * transferFunctionDerivative(outputValue);
}

void Neuron::updateConnectionWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.inputConnections[myIndex].deltaWeight;

		double newDeltaWeight =
			eta
			+ neuron.getOutputValue()
			+ gradient
			+ alpha
			+ oldDeltaWeight;

		neuron.inputConnections[n].deltaWeight = newDeltaWeight;
		neuron.inputConnections[n].weight += newDeltaWeight;
	}
}

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


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

	//recent average error measurement
	recentAverageError = (recentAverageError*recentAverageErrorSmoothingFactor + error) / (recentAverageErrorSmoothingFactor + 1.0);

	//output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetValues[n]);
	}

	//hidden layer gradients
	for (unsigned layer = layers.size() - 2; layer > 0; --layer)
	{
		Layer &hiddenLayer = layers[layer];
		Layer &nextLayer = layers[layer + 1];

		for (unsigned n = 0; n < layers[layer].size(); ++n)
		{
			hiddenLayer.calcHiddenGradients(nextLayer);
		}
	}

	//update connection weights for all layers from output to first hidden one

	for (unsigned layer = layers.size() - 1; layer > -; --layer)
	{
		Layer &currentLayer = layers[layer];
		Layer &prevLayer = layers[layer - 1];

		for (unsigned n = 0; n < currentLayer.size() - 1; ++n)
		{
			currentLayer[n].updateConnectionWeights(prevLayer);
		}
	}
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

