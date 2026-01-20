#ifndef COUNT_H
#define COUNT_H

#include "StreamingStatistic.h"

namespace Statistics {

//! An statistic to count the number of values
//template <typename FunctionType>
class Count : public StreamingStatistic
{
public:

  //! Make sure you overload the addVertex call
  using StreamingStatistic::add;

  //! Default constructor
  Count() : StreamingStatistic(), mCount(initialValue()) {}

  //Copy constructor
  Count(const Count& count);

  //! Destructor
  ~Count() {}

  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/

  //! Return the type of this statistic
  StreamingStatisticType type() const {return ATT_COUNT;}

  //! Return the type name 
  const char* typeName() const {return "count";}
  
  //! Return the initial value
  double initialValue() const { return 0.0; };
  
  //! Reinitialize this statistic
  void reset() { mCount = initialValue(); };

  //! Add another value to the statistic
  void add(double f);
  
   //! Aggregate another statistic
  void add(const StreamingStatistic* seg);

  //! Return the value as double
  double value() const {return mCount;}

  //! Return the value
  uint64_t count() const {return mCount;}

  //! Serialize the statistic
  void serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff = false) const;

  //! Deserialize the statistic
  void deserialize(int32_t buff_sz, const char* buff_ptr);
  
  /*************************************************************************************
   **************************** File Interface *****************************************
   ************************************************************************************/

  //! Write the ascii version of the element to the file
  void writeASCII(std::ofstream& input) const;

  //! Read the ascii version of the element from the file
  void readASCII(std::ifstream& input);


public:
  
  uint64_t mCount;

};

} // Closing the namespace

#endif
