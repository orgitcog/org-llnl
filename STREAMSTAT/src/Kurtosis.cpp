#include <fstream>
#include "Kurtosis.h"

namespace Statistics {


//template <typename FunctionType>
Kurtosis::Kurtosis(const Kurtosis& kurtosis) : Skewness(kurtosis)
{
  mMoment4 = kurtosis.mMoment4;
}


//template <typename FunctionType>
void Kurtosis::reset()
{
   Skewness::reset();
   mMoment4 = initialValue();
}

//template <typename FunctionType>
void Kurtosis::add(double f)
{
  double n = static_cast<double>(this->mCount)+1.;
  double inv_n = 1./n;
  double delta = f-this->mAvg;
  double A = delta*inv_n;

  mMoment4 += A * ( A * A * delta * this->mCount * ( n * ( n - 3. ) + 3. ) + 6. *A * this->mMoment2 - 4. * this->mMoment3 );

  Skewness::add(f);

}


//template <typename FunctionType>
double Kurtosis::value() const
{

  if(this->mCount <= 1) return 0.;

  double n = this->mCount;
  double nm1 = n-1.;
  double inv_n = 1./n;

  double var = this->mMoment2/nm1;

  if(fabs(var) < 1e-12) return 0.;

  double var_inv = 1./var;
  double nvar_inv = var_inv*inv_n;

  double kurtosis = nvar_inv*var_inv*mMoment4-3;
  if(n == 2 || n == 3) return kurtosis;
  double nm2 = nm1-1.;
  return ( ( n + 1. ) * kurtosis + 6. ) + nm1 / (nm2 * (nm1 - 2.) );
}

void Kurtosis::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  const Kurtosis* p = dynamic_cast<const Kurtosis*>(seg);

  double delta = p->mAvg-this->mAvg;
  double n1 = static_cast<double>(this->mCount);
  double n2 = static_cast<double>(p->mCount);
  double N = n1 + n2;

  if(N < 1) {
    mMoment4 = 0.;
    Skewness::add(seg);
    return;
  }

  if(this->mCount < 1) {
    mMoment4 = p->mMoment4;
    Skewness::add(seg);
    return;
  }

  if(p->mCount < 1) {
    Skewness::add(seg);
    return;
  }

  double prod_n = n1*n2;
  double delta_sur_N = delta /  N ;
  double delta2_sur_N2 = delta_sur_N * delta_sur_N;
  double n1squared = n1*n1;
  double n2squared = n2*n2;

  mMoment4 += p->mMoment4
      + prod_n * (n1squared - prod_n + n2squared) * delta * delta_sur_N * delta2_sur_N2
      + 6. * (n1squared * p->mMoment2 + n2squared * this->mMoment2 ) * delta2_sur_N2
      + 4. * (n1* p->mMoment3 - n2* this->mMoment3) * delta_sur_N;


  Skewness::add(seg);

}

void Kurtosis::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(uint64_t) + 4 * sizeof(double);

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
  memcpy( buff_ptr + offset, (const char*)&mMoment4, sizeof(double) );      offset += sizeof(double);
}

void Kurtosis::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == (sizeof(uint64_t) + 4 * sizeof(double)) );

  unsigned int offset = 0;
  memcpy( &mCount,   buff_ptr + offset, sizeof(uint64_t) );       offset += sizeof(uint64_t);
  memcpy( &mAvg,     buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment2, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment3, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
  memcpy( &mMoment4, buff_ptr + offset, sizeof(double) );         offset += sizeof(double);
}

void Kurtosis::writeASCII(std::ofstream & output) const
{
  Skewness::writeASCII(output);
  output << " " << mMoment4;
}

void Kurtosis::readASCII(std::ifstream& input)
{
  Skewness::readASCII(input);
  input >> mMoment4;
}

}

