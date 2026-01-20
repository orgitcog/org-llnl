#ifndef MINIMUM_H
#define MINIMUM_H

#include "StreamingStatistic.h"

namespace Statistics {

//! An statistic to record the minimal value
//template <typename FunctionType>
class Minimum : public StreamingStatistic
{
public:

  //! Make sure you overload the addVertex call
  using StreamingStatistic::add;

  //! Default constructor
  Minimum() : StreamingStatistic(), mMin(initialValue()) {}

  //Copy constructor
  Minimum(const Minimum& val);

  //! Destructor
   ~Minimum() {}

  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/
 
  //! Return the type of this statistic
   StreamingStatisticType type() const {return ATT_MIN;}

  //! Return a string identifying the statistic
  const char* typeName() const {return "min";}
 
  //! Reinitialize this statistic
   void reset() { mMin = initialValue(); };

  //! Return the initial value
   double initialValue() const { return gStatisticsMaxValue; }; 

  //! Add another value to the statistic
   void add(double f);
  
  //! Aggregate another statistic
   void add(const StreamingStatistic* seg);

  //! Return the value as double
   double value() const {return mMin;}
  
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
  
  //! The minimal function value seen so far
  double mMin;

};


} // Closing the namespace






#endif

