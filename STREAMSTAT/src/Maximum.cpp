#include <fstream>
#include "Maximum.h"

#include <algorithm>


namespace Statistics {


//template <typename FunctionType>
Maximum::Maximum(const Maximum& val) :
  StreamingStatistic(val)
{
  mMax = val.mMax;
}

void Maximum::add(double f)
{
  mMax = std::max(f,mMax);
}

void Maximum::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  mMax = std::max(mMax,seg->value());
}

void Maximum::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  memcpy( buff_ptr, (const char*)&mMax, buff_sz );
}

void Maximum::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == sizeof(double) );

  memcpy( &mMax, buff_ptr, buff_sz );
}

void Maximum::writeASCII(std::ofstream& output) const
{
  output << mMax << " ";
}

void Maximum::readASCII(std::ifstream& input)
{
  input >> mMax;
}

}

