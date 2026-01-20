#ifndef STATISTICFACTORY_H
#define STATISTICFACTORY_H

#include "string.h"
#include "Mean.h"
#include "Maximum.h"
#include "Minimum.h"
#include "Sum.h"
#include "Variance.h"
#include "Skewness.h"
#include "Kurtosis.h"
#include "Count.h"

namespace Statistics {


class Factory
{
public:

  //! Constructor
  Factory() {}

  ~Factory() {}


  StreamingStatistic* make_statistic(StreamingStatisticType tp) const;
  StreamingStatistic* make_statistic(const std::string& name) const { return make_statistic(type(name.c_str())); }
  StreamingStatistic* make_statistic(const char* name) const { return make_statistic(type(name)); }

private:

  Count mCount;
  Mean mMean;
  Maximum mMaximum;
  Minimum mMinimum;
  Sum mSum;
  Variance mVariance;
  Skewness mSkewness;
  Kurtosis mKurtosis;

  //! Get the type from the name
  StreamingStatisticType type(const char* name) const;
};
  


}


#endif
