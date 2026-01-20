#include <fstream>
#include "Skewness.h"

namespace Statistics {


Skewness::Skewness(const Skewness& skewness) : Variance(skewness)
{
  mMoment3 = skewness.mMoment3;
}

void Skewness::reset()
{
   Variance::reset();
   mMoment3 = initialValue();
}

void Skewness::add(double f)
{
  double n = static_cast<double>(this->mCount)+1;
  double inv_n = 1./n;
  double delta = f-this->mAvg;
  double A = delta*inv_n;
  double tempMean = this->mAvg + delta*inv_n;
  double B = f-tempMean;

  mMoment3 += A * ( B * delta * ( n - 2. ) - 3. * this->mMoment2 );

  Variance::add(f);

}

double Skewness::value() const
{

  if(this->mCount <= 1) return 0.;

  double n = this->mCount;
  double nm1 = n-1.;
  double inv_n = 1./n;

  double var = this->mMoment2/nm1;

  if(fabs(var) < 1e-12) return 0.;

  double var_inv = 1./var;
  double nvar_inv = var_inv*inv_n;
  double skew = nvar_inv*sqrt(var_inv)*mMoment3;
  if(n == 2) return skew;
  double nm2 = nm1-1.;
  return ( n * n ) / (nm1-nm2) * skew;


}

void Skewness::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  const Skewness* p = dynamic_cast<const Skewness*>(seg);

  double n1 = static_cast<double>(this->mCount);
  double n2 = static_cast<double>(p->mCount);
  double N = n1 + n2;
  if(N < 1) {
    mMoment3 = 0.;
    Variance::add(seg);
    return;
  }
  if(this->mCount < 1) {
    mMoment3 = p->mMoment3;
    Variance::add(seg);
    return;
  }
  if(p->mCount < 1) {
    Variance::add(seg);
    return;
  }

  double prod_n = n1*n2;
  double delta = p->mAvg-this->mAvg;
  double delta_sur_N = delta /  N ;

  mMoment3 += p->mMoment3
      + prod_n * (n1-n2) * delta * delta_sur_N * delta_sur_N
      + 3. * (n1*p->mMoment2 - n2*this->mMoment2 ) * delta_sur_N;

  Variance::add(seg);
}

void Skewness::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(uint64_t) + 3 * sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  unsigned int offset = 0;
  memcpy( buff_ptr + offset, (const char*)&mCount,   sizeof(uint64_t) );    offset += sizeof(uint64_t);
  memcpy( buff_ptr + offset, (const char*)&mAvg,     sizeof(double) );      offset += sizeof(double);
  memcpy( buff_ptr + offset, (const char*)&mMoment2, sizeof(double) );      offset += sizeof(double);
  memcpy( buff_ptr + offset, (const char*)&mMoment3, sizeof(double) );      offset += sizeof(double);
}

void Skewness::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == (sizeof(uint64_t) + 3 * sizeof(double)) );

  unsigned int offset = 0;
  memcpy( &mCount,   buff_ptr + offset, sizeof(uint64_t) );       offset += sizeof(uint64_t);
  memcpy( &mAvg,     buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment2, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment3, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
}

void Skewness::writeASCII(std::ofstream & output) const
{
  Variance::writeASCII(output);
  output << " " << mMoment3;
}

void Skewness::readASCII(std::ifstream& input)
{
  Variance::readASCII(input);
  input >> mMoment3;
}
}

