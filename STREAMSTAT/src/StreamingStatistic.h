#ifndef STREAMINGSTATISTIC_H
#define STREAMINGSTATISTIC_H

#include <cstdio>
#include <string>
#include <vector>
#include <math.h>
#include "StatisticsDefinitions.h"

namespace Statistics {

enum StreamingStatisticType {
  ATT_BASE  = 0,
  ATT_COUNT  = 1,
  ATT_MEAN = 2,
  ATT_MAX = 3,
  ATT_MIN = 4,
  ATT_SUM = 5,
  ATT_VAR = 6,
  ATT_SKEW = 7,
  ATT_KURT = 8,
};

class StreamingStatistic
{
public:

  //! Standard constructor
  StreamingStatistic() {}

  //! Destructor
  ~StreamingStatistic() {}


  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/

  //! Reset the statistic
  virtual void reset() {sterror(true,"Necessary function not overloaded.");}

  //! Return the initial value as a double
  virtual double initialValue() const {sterror(true,"Necessary function not overloaded.");return 0;}

  //! Add another value to the statistic
  virtual void add(double f) {sterror(true,"Necessary function not overloaded.");}

  //! Aggregate another statistic
  virtual void add(const StreamingStatistic* seg) {sterror(true,"Necessary function not overloaded.");}

  //! Return the value as double
  virtual double value() const {sterror(true,"Necessary function not overloaded.");return 0;}

  //! Return the type of this statistic 
  virtual StreamingStatisticType type() const { sterror(true,"Necessary function not overloaded."); return ATT_BASE;}

  //! Return the type name
  virtual const char* typeName() const { sterror(true,"Necessary function not overloaded.");return 0;}

  //! Serialize the statistic
  virtual void serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff = false) const { sterror(true,"Necessary function not overloaded."); }

  //! Deserialize the statistic
  virtual void deserialize(int32_t buff_sz, const char* buff_ptr) { sterror(true,"Necessary function not overloaded."); }
};

}

#endif
