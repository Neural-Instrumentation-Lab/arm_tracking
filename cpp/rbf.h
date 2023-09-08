/*
 * rbf.h
 *
 *  Created on: Jun 21, 2011
 *      Author: iobeid
 */

#ifndef _RBF
#define _RBF
class rbf{
 private:
  int nDimensions;
  double *center;
  double sigma;
 public:
  rbf();
  rbf(int,double*,double);
  ~rbf();
  void init(int,double*,double);
  double activation(double*);
  double getCenter(int);
};
#endif

