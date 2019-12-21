#pragma once
#include "Layer.h"

/*An outputLayer simply takes the output of the last hidden layer, and adds everything together.*/
class OutputLayer :
	public Layer
{
public:
	OutputLayer(int numberInputs);
	void calcOutput() override;
};

