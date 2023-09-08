/*
 * cerebellum.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: iobeid
 */

#include <iostream>
#include <iomanip>
#include "cerebellum.h"
using namespace std;

cerebellum::cerebellum(){
  nCenters = 0;
  cells = NULL;
  weights = NULL;
  lastActivation = NULL;
}

cerebellum::cerebellum(int n, double **ctrs, int nDims, double sg, double b){
  init(n,ctrs,nDims,sg, b);
}

void cerebellum::init(int n, double **ctrs, int nD, double sg, double b){
  int i;
  nCenters = n;
  nDims = nD;
  cells = new rbf[nCenters];
  for(i=0;i<nCenters;i++)
    cells[i].init(nDims,ctrs[i],sg);

  weights = new double[nCenters];
  lastActivation = new double [nCenters];

  for(i=0;i<nCenters;i++){
    weights[i] = 0;
    lastActivation[i] = 0;
  }

  beta = b;
}

double cerebellum::stimulate(double *pt){
  int i;
  double s = 0;
  for (i=0;i<nCenters;i++){
    lastActivation[i] = cells[i].activation(pt);
    s += lastActivation[i] * weights[i];
  }
  return s;
}

double cerebellum::getWeight(int i){
	return weights[i];
}

void cerebellum::updateErrors(double error){
  int i;
  for(i=0;i<nCenters;i++)
    weights[i] -= beta * error * lastActivation[i];
}

void cerebellum::reportStatus(){
  int i,j;
  cout << setiosflags(ios::fixed);
  cout << setprecision(2);

  for (i=0;i<nCenters;i++){
    cout << i << "\t";
    for (j=0;j<nDims;j++)
      cout << cells[i].getCenter(j) << "\t";
    cout << weights[i] << "\t";
    cout << lastActivation[i] << endl;
  }
}

cerebellum::~cerebellum(){
  delete [] cells;
  delete [] weights;
  delete [] lastActivation;
}
