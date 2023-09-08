/*
 * cerebellum.h
 *
 *  Created on: Jun 21, 2011
 *      Author: iobeid
 */

#ifndef _CEREBELLUM
#define _CEREBELLUM

#include "rbf.h"
#define _BETA 0.05

class cerebellum{
 private:
  double beta;
  int nDims;
  int nCenters;
  rbf *cells;
  double *weights;
  double *lastActivation;
 public:
  cerebellum();
  cerebellum(int,double**,int,double,double = _BETA);
  ~cerebellum();
  void init(int,double**,int,double,double = _BETA);
  double stimulate(double*);
  double getWeight(int);
  void updateErrors(double);
  void reportStatus();
};
#endif

