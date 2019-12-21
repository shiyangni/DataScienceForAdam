#include "HiddenLayer.h"

double HiddenLayer::sigmoid(double x)
{
	return 0.0;
}

HiddenLayer::HiddenLayer(int numberInputs, int numberOutputs)
{
}

void HiddenLayer::calcOutput()
{
}

Eigen::VectorXd HiddenLayer::activation(Eigen::VectorXd)
{
	return Eigen::VectorXd();
}

Eigen::MatrixXd HiddenLayer::getWeights()
{
	return Eigen::MatrixXd();
}

void HiddenLayer::setWeights(Eigen::MatrixXd)
{
}

Eigen::VectorXd HiddenLayer::getJthWeight()
{
	return Eigen::VectorXd();
}

Eigen::VectorXd HiddenLayer::getBias()
{
	return Eigen::VectorXd();
}

void HiddenLayer::setBias(Eigen::VectorXd)
{
}

Eigen::VectorXd HiddenLayer::getNeblaWeights()
{
	return Eigen::VectorXd();
}

void HiddenLayer::setNeblaWeights()
{
}

Eigen::VectorXd HiddenLayer::getNeblaBias()
{
	return Eigen::VectorXd();
}

void HiddenLayer::setNeblaBias()
{
}
