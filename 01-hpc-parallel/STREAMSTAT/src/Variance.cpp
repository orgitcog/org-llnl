#include <fstream>
#include "Variance.h"

namespace Statistics {



Variance::Variance(const Variance& variance) : Mean(variance)
{
  mMoment2 = variance.mMoment2;
}

void Variance::reset()
{
   Mean::reset();
   mMoment2 = initialValue();
}

void Variance::add(double f)
{
  double delta= f-this->mAvg;

  Mean::add(f);

  mMoment2 +=  delta*(f-this->mAvg);

}

double Variance::value() const
{
  if (this->mCount <= 1) return 0.;
  double inv_nm1 = 1./static_cast<double>(this->mCount);
  return (mMoment2*inv_nm1);

}


void Variance::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  const Variance* p = dynamic_cast<const Variance*>(seg);

  double delta = p->mAvg-this->mAvg;
  int N = this->mCount + p->mCount;
  if(N < 1) {
    mMoment2 = 0.;
    Mean::add(seg);
    return;
  }

  if(this->mCount < 1) {
    mMoment2 = p->mMoment2;
    Mean::add(seg);
    return;
  }
  if(p->mCount < 1) {
    Mean::add(seg);
    return;
  }

  int prod_n = this->mCount*p->mCount;
  double delta_sur_N = delta / static_cast<double>( N );

  mMoment2 += p->mMoment2 + prod_n*delta*delta_sur_N;

  Mean::add(seg);

}

void Variance::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(uint64_t) + 2 * sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];

  unsigned int offset = 0;
  memcpy( buff_ptr + offset, (const char*)&mCount,   sizeof(uint64_t) );    offset += sizeof(uint64_t);
  memcpy( buff_ptr + offset, (const char*)&mAvg,     sizeof(double) );      offset += sizeof(double);
  memcpy( buff_ptr + offset, (const char*)&mMoment2, sizeof(double) );      offset += sizeof(double);
}

void Variance::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == (sizeof(uint64_t) + 2 * sizeof(double)) );

  unsigned int offset = 0;
  memcpy( &mCount,   buff_ptr + offset, sizeof(uint64_t) );       offset += sizeof(uint64_t);
  memcpy( &mAvg,     buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment2, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
}

void Variance::writeASCII(std::ofstream& output) const
{
  Mean::writeASCII(output);
  output << " " << mMoment2;// << " " << value();
}

void Variance::readASCII(std::ifstream& input)
{
  Mean::readASCII(input);
  input >> mMoment2;
}
}

