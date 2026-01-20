#ifndef Mean_H
#define Mean_H

#include <iostream>
#include "Count.h"

namespace Statistics {

//template<typename FunctionType>
class Mean : public Count
{
public:

  //! Make sure you overload the addVertex call
  using StreamingStatistic::add;

  //! Default constructor
  Mean() : Count(), mAvg(initialValue()) {}
  
  //! Copy constructor
  Mean(const Mean& avg);

  ~Mean() {}

  //! Return the type of this attribute
  StreamingStatisticType type() const {return ATT_MEAN;}

  //! Return a string identifying the attribute
  const char* typeName() const {return "mean";}
 
  //! Reinitialize the statistic
  void reset();

  //! Return the initial value
  double initialValue() const { return 0.0; }; 

  //! Add another value to the statistic
  void add(double f);
  
  //! Aggregate another statistic
  void add(const StreamingStatistic* seg);

  //! Return the value as double
  double value() const {return mAvg;}

  //! Return the function values
  double mean() const {return mAvg;}

  //! Serialize the statistic
  void serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff = false) const;

  //! Deserialize the statistic
  void deserialize(int32_t buff_sz, const char* buff_ptr);

  /*************************************************************************************
   **************************** File Interface *****************************************
   ************************************************************************************/
  
  //! Write the ascii version of the element to the file
  void writeASCII(std::ofstream& output) const;

  //! Read the ascii version of the element from the file
  void readASCII(std::ifstream& input);
public:

  double mAvg;

protected:

  //! Add the given count and MeanBase to the current MeanBase
  void addRawValues(uint64_t count, double MeanBase);
  
};

} // Closing the namespace

  
#endif
