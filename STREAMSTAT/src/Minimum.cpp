#include <fstream>
#include "Minimum.h"

#include <algorithm>


namespace Statistics {


//template <typename FunctionType>
Minimum::Minimum(const Minimum& val) :
  StreamingStatistic(val)
{
  mMin = val.mMin;
}

void Minimum::add(double f)
{
  mMin = std::min(f,mMin);
}

void Minimum::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  mMin = std::min(mMin,seg->value());
}

void Minimum::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  memcpy( buff_ptr, (const char*)&mMin, buff_sz );
}

void Minimum::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == sizeof(double) );

  memcpy( &mMin, buff_ptr, buff_sz );
}

void Minimum::writeASCII(std::ofstream& output) const
{
  output << mMin << " ";
}

void Minimum::readASCII(std::ifstream& input)
{
  input >> mMin;
}

}

