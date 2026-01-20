#ifndef SKEWNESS_H
#define SKEWNESS_H

#include <math.h>
#include "Variance.h"


/***************************************************************************
* This code makes use of both incremental and pairwise update formulas
* from "Numerically Stable, Single-Pass, Parallel Statisitcs Algorithms"
* available online at 
* http://www.janinebennett.org/index_files/ParallelStatisticsAlgorithms.pdf
* The paper has derivations for arbitrary-order centered moments as well as 
* for covariance.
****************************************************************************/

namespace Statistics {

class Skewness : public Variance
{
public:

  //! Make sure you overload the addVertex call
  using Variance::add;

  //! Default constructor
  Skewness() : Variance(), mMoment3(initialValue()) {}
  
  //! Copy constructor
  Skewness(const Skewness& skewness);

   ~Skewness() {}

  /*************************************************************************************
   **************************** Computation Interface **********************************
   ************************************************************************************/

  //! Return the type of this attribute
  StreamingStatisticType type() const {return ATT_SKEW;}

  //! Return a string identifying the attribute
  const char* typeName() const {return "skewness";}
  
  //! Reinitialize the statistic
  void reset();

  //! Return the initial value
  double initialValue() const { return 0.0; };

  //! Add another value to the statistic
  void add(double f);
  
  //! Aggregate another statistic
  void add(const StreamingStatistic* seg);

  //! Return the value as double
  double value() const;
  
  //! Serialize the statistic
  void serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff = false) const;

  //! Deserialize the statistic
  void deserialize(int32_t buff_sz, const char* buff_ptr);

  //! Write the ascii version of the element to the file
  void writeASCII(std::ofstream& output) const;

  //! Read the ascii version of the element from the file
  void readASCII(std::ifstream& input);

public:

  double mMoment3;

};

} // Closing the namespace

  
#endif
