#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include <vector>
#include <algorithm>

#include "StreamingStatisticFactory.h"

using namespace std;
using namespace Statistics;

#define TEST_COUNT  8
const char* gTestCases[TEST_COUNT] = {
    "count",
    "min",
    "max",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "sum",
};


int main(int argc, const char** argv)
{
  uint64_t total = 1024;
  vector<double> values(total);
  vector<double>::iterator it;
  vector<uint64_t> order(total);

  Factory fac;

  /* initialize random seed: */
  srand (42);

  // Pull some random numbers
  for (it=values.begin();it!=values.end();it++)
    *it = rand() % 100;

  // Initialize the order
  for (int i=0;i<total;i++)  {
    order[i] = i;
  }
  random_shuffle(order.begin(),order.end());

  // For all the test cases
  for (int i=0;i<TEST_COUNT;i++) {

    StreamingStatistic* stat1 = fac.make_statistic(gTestCases[i]);
    StreamingStatistic* stat2 = fac.make_statistic(gTestCases[i]);

    vector<StreamingStatistic*> stat3(total,NULL);

    for (int j=0;j<total;j++) {
      stat1->add(values[j]);
      stat2->add(values[order[j]]);

      stat3[j] = fac.make_statistic(gTestCases[i]);
      stat3[j]->add(values[j]);
    }

    for (int step=1;step<total;step*=2) {
      for (int k=0;k<total;k+=2*step)
        stat3[k]->add(stat3[k+step]);
    }

    cout << gTestCases[i] << ":  Original=" << stat1->value() << "  Reorder=" << stat2->value() << "  Hierarchical=" << stat3[0]->value() << endl;

    // Test serialization
    char* buff_ptr = nullptr;
    int32_t buff_sz = 0;
    stat1->serialize( buff_sz, buff_ptr, true );

    {
      StreamingStatistic* temp_stat = fac.make_statistic(gTestCases[i]);
      temp_stat->deserialize( buff_sz, buff_ptr );
      assert( temp_stat->value() == stat1->value() );
      delete temp_stat;
    }

    delete[] buff_ptr;  buff_ptr = nullptr;

    stat1->serialize( buff_sz, buff_ptr, false );
    buff_ptr = new char[buff_sz];
    stat1->serialize( buff_sz, buff_ptr, false );

    {
      StreamingStatistic* temp_stat = fac.make_statistic(gTestCases[i]);
      temp_stat->deserialize( buff_sz, buff_ptr );
      assert( temp_stat->value() == stat1->value() );
      delete temp_stat;
    }

    delete[] buff_ptr;  buff_ptr = nullptr;
  }

  return 0;
}
