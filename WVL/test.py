#/usr/bin/env Priithon 
from Priithon import *
#import Wvl
from Priithon import Wvl

expected_output = '''\
setting numpy.radom.seed(42)
a.shape=(1, 256, 256) a.dtype=float32 U.mmms(a)=(0.0, 8.0, 0.9979248046875, 0.9962217884424833)
b.shape=(1, 256, 256) b.dtype=float32 U.mmms(b)=(0.0, 0.0, 0.0, 0.0)
begin angle 0
begin angle  15.0
begin angle  30.0
begin angle  45.0
begin angle  60.0
begin angle  75.0
b.shape=(1, 256, 256) b.dtype=float32 U..mmms(b)=(0.0, 0.870481014251709, 0.06549153077987899, 0.08084528903298893\
'''
print '---------------  expected output    -----------------------------------'
print expected_output
print '-----------------------------------------------------------------------'
N.random.seed(42)
print 'setting numpy.radom.seed(42)'
a=F.poissonArr(shape=(1,256,256),mean=1,dtype=N.float32)
b=F.zeroArrF(a.shape)
print("a.shape=%s a.dtype=%s U.mmms(a)=%s"%(a.shape, a.dtype, U.mmms(a)))
print("b.shape=%s b.dtype=%s U.mmms(b)=%s"%(b.shape, b.dtype, U.mmms(b)))
Wvl.twoD(a,b,4)
print("b.shape=%s b.dtype=%s U..mmms(b)=%s"%(b.shape, b.dtype, U.mmms(b)))
