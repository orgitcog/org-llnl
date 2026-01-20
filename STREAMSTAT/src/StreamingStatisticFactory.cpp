#include <cstring>
#include "StreamingStatisticFactory.h"

namespace Statistics {

StreamingStatistic* Factory::make_statistic(StreamingStatisticType tp) const
{
  StreamingStatistic *arr = NULL;
  switch (tp) {
    case ATT_COUNT:
      arr = new Count();
      break;
    case ATT_MEAN:
      arr = new Mean();
      break;
    case ATT_MAX:
      arr = new Maximum();
      break;
    case ATT_MIN:
      arr = new Minimum();
      break;
    case ATT_SUM:
      arr = new Sum();
      break;
    case ATT_VAR:
      arr = new Variance();
      break;
    case ATT_SKEW:
      arr = new Skewness();
      break;
    case ATT_KURT:
      arr = new Kurtosis();
      break;
    default:
      sterror(true, "Unknown streaming statistics type  %d", tp);
      break;
  }

  return arr;
}

StreamingStatisticType Factory::type(const char* name) const
{
  /*
  std::cout << "debug print cur name:" << name << std::endl;
  std::cout << "debug print Value:" << mValueArray.typeName() << std::endl;
  std::cout << "debug print VertexCount:" << mVertexCountArray.typeName() << std::endl;
  std::cout << "debug print mean:" << mMeanArray.typeName() << std::endl;
  std::cout << "debug print maximum:" << mMaximalFunctionValueArray.typeName() << std::endl;
  std::cout << "debug print minimum:" << mMinimalFunctionValueArray.typeName() << std::endl;
  std::cout << "debug print sum:" << mSumArray.typeName() << std::endl;
  std::cout << "debug print variance:" << mFunctionVarianceArray.typeName() << std::endl;
  std::cout << "debug print skewness:" << mFunctionSkewnessArray.typeName() << std::endl;
  std::cout << "debug print kurtosis:" << mFunctionKurtosisArray.typeName() << std::endl;
  std::cout << "debug print covariance:" << mCovarianceArray.typeName() << std::endl;
  std::cout << "debug print weighted mean:" << mWeightedMeanArray.typeName() << std::endl;
  std::cout << "debug print periodicmean:" << mPeriodicMeanArray.typeName() << std::endl;
  */

  if (strcmp(name,mCount.typeName())==0) {
    return ATT_COUNT;
  }
  else if (strcmp(name, mMean.typeName())==0) {
    return ATT_MEAN;
  }
  else if (strcmp(name,mMaximum.typeName())==0) {
    return ATT_MAX;
  }
  else if (strcmp(name,mMinimum.typeName())==0) {
    return ATT_MIN;
  }
  else if (strcmp(name,mSum.typeName())==0) {
    return ATT_SUM;
  }
  else if (strcmp(name,mVariance.typeName())==0) {
    return ATT_VAR;
  }
  else if (strcmp(name,mSkewness.typeName())==0) {
    return ATT_SKEW;
  }
  else if (strcmp(name,mKurtosis.typeName())==0) {
    return ATT_KURT;
  }
  else {
    fprintf(stderr,"StreamingStatistic typename %s is not recognized",name);
    assert (false);
  }

  return ATT_COUNT;
}


}

