#ifndef MAXIMUM_H
#define MAXIMUM_H

#include "StreamingStatistic.h"

namespace Statistics {

//! An statistic to record the maximal value
//template <typename FunctionType>
class Maximum : public StreamingStatistic
{
public:

  //! Make sure you overload the add call
  using StreamingStatistic::add;

  //! Default constructor
  Maximum() : StreamingStatistic(), mMax(initialValue()) {}

  //Copy constructor
  Maximum(const Maximum& val);

  //! Destructor
  ~Maximum() {}

  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/

  //! Return the type of this attribute
  StreamingStatisticType type() const {return ATT_MAX;}

  //! Return a string identifying the attribute
  const char* typeName() const {return "max";}

  //! Return the initial value
  double initialValue() const { return gStatisticsMinValue; }; 

  //! Reinitialize the statistic
  void reset() { mMax = initialValue(); };

  //! Add another value to the statistic
  void add(double f);
  
  //! Aggregate another statistic
  void add(const StreamingStatistic* seg);

  //! Return the value as double
  double value() const {return mMax;}
  
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
  
  //! The maximal function value seen so far
  double mMax;
};

} // Closing the namespace






#endif

