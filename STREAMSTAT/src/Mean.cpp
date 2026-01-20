#include <fstream>
#include "Mean.h"

namespace Statistics {


//template <typename FunctionType>
Mean::Mean(const Mean& avg) : Count(avg)
{
  mAvg = avg.mAvg;
}


void Mean::reset()
{
  Count::reset();
  mAvg = initialValue();
}

void Mean::add(double f)
{
  int n = this->mCount+1;
  double inv_n = 1./static_cast<double>(n);
  double delta = f-mAvg;
  double A = delta*inv_n;
  mAvg += A;

  Count::add(f);
}

void Mean::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  const Mean* p = dynamic_cast<const Mean*>(seg);

  // This new variant of the MeanBase computation is necessary for the periodic
  // computation
  addRawValues(p->mCount,p->mAvg);
  return;


  double delta = p->mAvg - this->mAvg;
  int N = this->mCount + p->mCount;

  if(N < 1) {
    mAvg = 0.;
    return;
  }

  if(this->mCount < 1) {
    mAvg = p->mAvg;
    Count::add(seg);
    return;
  }

  if(p->mCount < 1) {
    Count::add(seg);
    return;
  }

  double delta_sur_N = delta / static_cast<double>( N );

  mAvg += p->mCount*delta_sur_N;

  Count::add(seg);

}

void Mean::addRawValues(uint64_t count, double MeanBase)
{
  double delta = MeanBase - this->mAvg;
  int N = this->mCount + count;

  if(N < 1) {
    mAvg = 0.;
    return;
  }

  if(count < 1)
     return;

  if(this->mCount < 1) {
    mAvg = MeanBase;
    this->mCount += count;
    return;
  }


  double delta_sur_N = delta / static_cast<double>( N );

  mAvg += count*delta_sur_N;
  this->mCount += count;
}

void Mean::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(uint64_t) + sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  memcpy( buff_ptr, (const char*)&mCount, sizeof(uint64_t) );
  memcpy( buff_ptr + sizeof(uint64_t), (const char*)&mAvg, sizeof(double) );
}

void Mean::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == (sizeof(uint64_t) + sizeof(double)) );

  memcpy( &mCount, buff_ptr, sizeof(uint64_t) );
  memcpy( &mAvg, buff_ptr + sizeof(uint64_t), sizeof(double) );
}

void Mean::writeASCII(std::ofstream& output) const
{
  Count::writeASCII(output);
  output << " " << mAvg;
}

void Mean::readASCII(std::ifstream& input)
{
  Count::readASCII(input);
  input >> mAvg;
}

}

