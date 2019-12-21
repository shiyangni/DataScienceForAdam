#pragma once
#include <Eigen/Dense>
#include <vector>
#include "Layer.h"

/* A hidden layer has 
- a weighting matrix, and the associated derivative matrix. numOutputs X numInputs dimension.
- a bias vector, and the associated derivative matrix, numOutputs X 1 dimension.
- an activation function. We implement it as a sigmoid function. 

The neblas are the final derivatives used in sample_wise gradient descent. This means 
1. It's the averaged derivative dloss/dweights over all rows. These summand derivatives need to be stored somewhere.
2. It's only calculatable from the model. The fields here only act as storage. 


How are the neblas calculated?
For EACH SAMPLE, dloss/dw_kj = do_k/dw_kj * do_(k+1)/do_k * ... * do_L/do_(L-1) * do_final/do_L
(here the subscript k can be omitted since we're operating within one layer)*/
class HiddenLayer :
	public Layer
{
private:
	Eigen::MatrixXd weights;
	Eigen::VectorXd bias;
	/*This stores the do_(k+1)/do_k * ... * do_L/do_(L-1) * do_final/do_L for the current sample.
	The dimension is n_k X 1, or numOutputs X 1.
	Only calculatable from model.*/
	Eigen::VectorXd currSample_chainRuleMulFactor; 
	/*There are supposed to be n_k(numOutputs) elements in the vector.
	The jth element of the below vector is doutput/dw_j. The dimension for each
	element is p_k X n_k (numInputs X numOutputs). 
	Calculatable from this Layer.*/
	std::vector<Eigen::MatrixXd> currSample_dodweights;
	/*The jth row of the below matrix is nothing but the transpose of the jth element of the
	currSample_dodweights times currSample_chainRuleFactor*/
	Eigen::MatrixXd currSample_nebulaWeights;

	/*The vector contains n_k elements.
	The j-th element is nothing but doutput/dbias_j. The dimension is 1 X n_k.*/
	std::vector<Eigen::MatrixXd> currSample_dodbias;
	/*The jth row of the matrix below is the transpose of */

	/*These two vectors store the sample-wise nebla. The final nebla weights and bias
	are nothing but the average. Each of these elements*/
	std::vector<Eigen::MatrixXd> neblaWeights_bySampleVec;
	std::vector<Eigen::VectorXd> neblaBias_bySampleVec;

	Eigen::MatrixXd neblaWeights;
	Eigen::VectorXd neblaBias;
	double sigmoid(double x);

public:
	HiddenLayer(int numberInputs, int numberOutputs);

	void calcOutput() override;

	Eigen::VectorXd activation(Eigen::VectorXd);

	Eigen::MatrixXd getWeights();
	void setWeights(Eigen::MatrixXd);
	
	/* Get the weights in the jth neuron, and return it as a column vector.
	Notice this is just getting the jth row of the weighting matrix, and
	returning its tranposed version. */
	Eigen::VectorXd getJthWeight();


	Eigen::VectorXd getBias();
	void setBias(Eigen::VectorXd);

	Eigen::VectorXd getNeblaWeights();
	void setNeblaWeights();

	Eigen::VectorXd getNeblaBias();
	void setNeblaBias();


};

