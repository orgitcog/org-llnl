#include <fstream>
#include <cassert>
#include "Count.h"

namespace Statistics {


Count::Count(const Count& count) : StreamingStatistic(count)
{
  mCount = count.mCount;
}

void Count::add(double f)
{
  mCount++;
}

void Count::add(const StreamingStatistic* seg)
{
  sterror(type() != seg->type(), "Can only aggregate statistics of idenical type.");

  mCount += seg->value();
}

void Count::serialize(int32_t& buff_sz, char*& buff_ptr, bool create_buff) const
{
  buff_sz = sizeof(uint64_t);

  assert( !(buff_ptr && create_buff) );
  if( !buff_ptr && !create_buff )
    return;

  if( create_buff )
    buff_ptr = new char[buff_sz];
    
  memcpy( buff_ptr, (const char*)&mCount, buff_sz );
}

void Count::deserialize(int32_t buff_sz, const char* buff_ptr)
{
  assert( buff_sz == sizeof(uint64_t) );

  memcpy( &mCount, buff_ptr, buff_sz );
}

void Count::writeASCII(std::ofstream& output) const
{
  output << mCount;
}

//template <typename FunctionType>
void Count::readASCII(std::ifstream& input)
{
  input >> mCount;
}

}
