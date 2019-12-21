#include <Eigen/Dense>
#include "OutputLayer.h"

using namespace Eigen;

OutputLayer::OutputLayer(int numberInputs)
{
	Layer(numberInputs, numberInputs);
	setNumInputs(numberInputs);
	setNumOutputs(1);
}

void OutputLayer::calcOutput()
{
	int numRows = getInput().rows();
	VectorXd ones(numRows);
	ones.fill(1);
	MatrixXd result = ones.transpose() * getInput();
	setOutput(result);
}
