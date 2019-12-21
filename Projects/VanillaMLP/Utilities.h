#pragma once

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <istream>
#include <vector>
#include <string>  



/*Reads a CSV file into a vector of vector of strings.*/
std::vector<std::vector<std::string>> csvToVecOfVecString(std::string fileName);

/*Reads a CSV file into a matrix. */
Eigen::MatrixXd readCSV(std::string fileName);

/*Numerically differentiate an vector input, vector output function, at a given input value.
The output should have dimension inputDim X funcDim.*/
Eigen::MatrixXd numericDiff(Eigen::VectorXd input, Eigen::VectorXd func(Eigen::VectorXd), double perturbance=0.000001);


