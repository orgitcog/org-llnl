#include <fstream>
#include "Sum.h"

namespace Statistics {

Sum::Sum(const Sum& avg) : StreamingStatistic(avg)
{
  mIntegral = avg.mIntegral;
}

void Sum::add(double f)
{
  mIntegral += f;
}

void Sum::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  mIntegral += seg->value();
}

void Sum::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(double);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  memcpy( buff_ptr, (const char*)&mIntegral, buff_sz );
}

void Sum::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == sizeof(double) );

  memcpy( &mIntegral, buff_ptr, buff_sz );
}

void Sum::writeASCII(std::ofstream& output) const
{
  output << mIntegral;
}

void Sum::readASCII(std::ifstream& input)
{
  input >> mIntegral;
}

}

