#ifndef SUM_H
#define SUM_H

#include "StreamingStatistic.h"

namespace Statistics {

class Sum : public StreamingStatistic
{
public:

  //! Make sure you overload the addVertex call
  using StreamingStatistic::add;

  //! Default constructor
  Sum() : StreamingStatistic(), mIntegral(initialValue()) {}
  
  //! Copy constructor
  Sum(const Sum& avg);

  ~Sum() {}

  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/

  //! Return the type of this statistic
  StreamingStatisticType type() const {return ATT_SUM;}

  //! Return the type name
  const char* typeName() const {return "sum";}

  //! Return the initial value
  double initialValue() const { return 0.0; }; 

  //! Reinitialize the statistic
  void reset() { mIntegral = initialValue(); };

  //! Add another value to the statistic
  void add(double f);
  
  //! Aggregate another statistic
  void add(const StreamingStatistic* seg);

  //! Return the value as double
  double value() const {return mIntegral;}

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

  //! The sum  of all function values seen so far
  double mIntegral;

};


} // Closing the namespace

  
#endif
