from __future__ import absolute_import
'''notes 2015:
(Pdb) p W.fastwv6(*( (tmpTrans, aWV) + ok1 ))
   *** NotImplementedError: NotImplementedError("Wrong number or type of arguments for overloaded function 'fastwv6'.\n  Possible C/C++ prototypes are:\n    fastwv6(float *,int,int,int,float *,int,int,int,int,int,int,bool const)\n    fastwv6(float *,int,int,int,float *,int,int,int,int,int,int)\n",)
   (Pdb) p ok1
   (4, 4, 1)
   (Pdb) p type(ok1[0])
   <type 'numpy.int64'>
   ==========================>  must be `int` not `numpy.int64` -- inserted `  map(int, <order-tuple> )`  at many places (search for `take`)


TypeError: Array must be contiguous.  A discontiguous array was given
   ==========================>  add .copy()             tmpTrans = N.transpose(a, _perm[i]).copy() # force contiguous!

'''
_perm = (
(0,1,2),
(0,2,1),
(1,0,2),
(2,1,0),
(2,0,1),
(1,2,0),
)
_invperm = (
(0,1,2),
(0,2,1),
(1,0,2),
(2,1,0),
(1,2,0),
(2,0,1),
)
_perm2d = (
(0,1,2),
(0,2,1)
)
_invperm2d = _perm2d

from .all import N, Y, U, F
from Priithon_bin import willy as W

def threeD(a, res, o, nAngles=2, verboseLvl=1):

    if o < 1:
        raise ValueError, "order need to >= 1"
    if nAngles < 1:
        raise ValueError, "nAngles need to >= 1"

    if verboseLvl >=1:
        import time
        startTime = time.clock()
    res[:] = 0
    
    aWV = F.zeroArrF(a.shape)
    aRot= F.zeroArrF(a.shape)
    c  = F.zeroArrF(a.shape)

    angles = N.arange(90./nAngles,  90., 90./nAngles)
    nOffAngles = len(angles) # = nAngles -1

    # do first fastwv6 on original data
    if verboseLvl >=1:
        print "begin angle 0"
        if verboseLvl >=2:
            print "     perm ",
        Y.refresh()

    for i in range(len(_perm)):
        if verboseLvl >=2:
            print i, 
            Y.refresh()
            
        tmpTrans = N.transpose(a, _perm[i]).copy() # force contiguous!
        aWV.shape = tmpTrans.shape
        W.fastwv6(tmpTrans,aWV, o,o,o, verboseLvl>=3)
        tmpTrans = N.transpose(aWV,_invperm[i])
        tmpTrans /= len(_perm) * (1 + (nOffAngles * 3))
        res += tmpTrans

    # now do it for a bunch of angles
    for angle in angles:
        if verboseLvl >=1:
            print "begin angle ",angle
            print "   axis:",
            Y.refresh()

        for axis in (0,1,2):
            if verboseLvl >=1:
                if verboseLvl >=2 and axis > 0:
                    print
                    print "   axis:",
                print axis, 
                Y.refresh()
            U.rot3d(a,aRot, angle, axis)

            if verboseLvl >=2:
                print
                print "     perm ",

            c[:] = 0
            for i in range(len(_perm)):
                #dbg  print "     perm:", i
                if verboseLvl >=2:
                    print i, 
                    Y.refresh()

                tmpTrans = N.transpose(aRot, _perm[i]).copy() # force contiguous!
                aWV.shape = tmpTrans.shape
                W.fastwv6(tmpTrans,aWV, o,o,o, verboseLvl>=3)
                tmpTrans  = N.transpose(aWV,_invperm[i])
                tmpTrans /= len(_perm) * (1 + (nOffAngles * 3))
                c += tmpTrans

            U.rot3d(c,aRot,-angle,axis)
            res += aRot

        if verboseLvl >=1:
                print
    if verboseLvl >=1:
        import time
        d = time.clock()-startTime
        print "time: %.2f secs - %.2f min - %.2f h " % (d, d/60., d/60./60.)

def threeD_MM(a, res, o, nAngles=2, verboseLvl=1, tmp='/tmp/'):

    import Mrc
    
    if o < 1:
        raise ValueError, "order need to >= 1"
    if nAngles < 1:
        raise ValueError, "nAngles need to >= 1"

    if verboseLvl >=1:
        import time
        startTime = time.clock()
    res[:] = 0
    
    #    aWV = Mrc.bindNew(tmp+"wvl_aWV", N.float32, a.shape)
    #    aRot= Mrc.bindNew(tmp+"wvl_aRot", N.float32, a.shape)
    #    c  =  Mrc.bindNew(tmp+"wvl_c", N.float32, a.shape)
    aWV = F.zeroArrF(a.shape)

    angles = N.arange(90./nAngles,  90., 90./nAngles)
    nOffAngles = len(angles) # = nAngles -1

    # do first fastwv6 on original data
    if verboseLvl >=1:
        print "begin angle 0"
        if verboseLvl >=2:
            print
            print "     perm "
        Y.refresh()

    for i in range(len(_perm)):
        if verboseLvl >=2:
            print "        ", i
            Y.refresh()
        
        tmpTrans = N.transpose(a, _perm[i]).copy() # force contiguous!
        aWV.shape = tmpTrans.shape
        W.fastwv6(tmpTrans,aWV, o,o,o, verboseLvl>=3)
        tmpTrans = N.transpose(aWV,_invperm[i])
        tmpTrans /= len(_perm) * (1 + (nOffAngles * 3))
        res += tmpTrans

    # now do it for a bunch of angles
    if len(angles):
        c  =  Mrc.bindNew(tmp+"wvl_c", N.float32, a.shape)
    for angle in angles:
        
        for axis in (0,1,2):
            if verboseLvl >=1:
                print "begin angle ",angle
                print "   axis:", axis
                Y.refresh()

            aRot = aWV
            U.rot3d(a,aRot, angle, axis)

            if verboseLvl >=2:
                print "     perm "

            c[:] = 0
            for i in range(len(_perm)):
                #dbg  print "     perm:", i
                if verboseLvl >=2:
                    print "        ", i
                    Y.refresh()

                tmpTrans = N.transpose(aRot, _perm[i]).copy() # force contiguous!
                aWV.shape = tmpTrans.shape
                W.fastwv6(tmpTrans,aWV, o,o,o, verboseLvl>=3)
                tmpTrans  = N.transpose(aWV,_invperm[i])
                tmpTrans /= len(_perm) * (1 + (nOffAngles * 3))
                c += tmpTrans

            U.rot3d(c,aRot,-angle,axis)
            res += aRot

        if verboseLvl >=1:
                print
    if verboseLvl >=1:
        import time
        d = time.clock()-startTime
        print "time: %.2lf secs - %.2lf min - %.2lf h " % (d, d/60., d/60./60.)



def twoD_arr_oldPleaseCompareWithNew(a,res, o, nAngles=6, verboseLvl=1):

    if o < 1:
        raise ValueError, "order need to >= 1"
    if nAngles < 1:
        raise ValueError, "nAngles need to >= 1"

    res[:]=0
    # following block of code converts 2D data to 3D,
    #so original twoD coding can be used
    if len(a.shape) == 2:
        a = a.view()
        a.shape = (1,) + a.shape
    if len(res.shape) == 2:
        res = res.view()
        res.shape = (1,) + res.shape

    aWV = F.zeroArrF(a.shape)
    aRot= F.zeroArrF(a.shape)
    c  = F.zeroArrF(a.shape)

    # do first wave3dChp on original data
    if verboseLvl >=1:
        print "begin angle 0"
        Y.refresh()
        
    for i in range(len(_perm2d)):
        tmpTrans = N.transpose(a, _perm2d[i]).copy() # force contiguous!
        aWV.shape = tmpTrans.shape
        W.fastwv6(tmpTrans, aWV, o,o,1, verboseLvl>=2)
        tmpTrans = N.transpose(aWV, _invperm2d[i])
        tmpTrans /= len(_perm2d) * nAngles
        res += tmpTrans

    # now do it for a bunch of angles
    # angles = range(30,90,  30) # won't include 90 ...

    angles = N.arange(90./nAngles,  90., 90./nAngles)

    for angle in angles:
        if verboseLvl >=1:
            print "begin angle ",angle
            Y.refresh()

        U.rot3d(a,aRot, angle, 0)
        c[:]=0

        for i in range(len(_perm2d)):

            tmpTrans = N.transpose(aRot, _perm2d[i]).copy() # force contiguous!
            aWV.shape = tmpTrans.shape
            W.fastwv6(tmpTrans,aWV, o,o,1, verboseLvl>=2)
            tmpTrans  = N.transpose(aWV, _invperm2d[i])
            tmpTrans /= len(_perm2d) * nAngles
            c += tmpTrans

        U.rot3d(c,aRot, -angle, 0)
        res += aRot



def twoD(a,res, o, nAngles=6, verboseLvl=1):
    twoD_OK(a,res, o, o, nAngles=nAngles, verboseLvl=verboseLvl)

def twoD_OK(a,res, o, k, nAngles=1, verboseLvl=1):
    
    if o < 1:
        raise ValueError, "order need to >= 1"
    if k < 1:
        raise ValueError, "order need to >= 1"
    if nAngles < 1:
        raise ValueError, "nAngles need to >= 1"

    res[:]=0 
    # following block of code converts 2D data to 3D,
    #so original twoD coding can be used
    if len(a.shape) == 2:
        a = a.view()
        a.shape = (1,) + a.shape
    if len(res.shape) == 2:
        res = res.view()
        res.shape = (1,) + res.shape

    aWV = F.zeroArrF(a.shape)
    aRot= F.zeroArrF(a.shape)
    c  = F.zeroArrF(a.shape)

    # do first wave3dChp on original data
    if verboseLvl >=1:
        print "begin angle 0"
        Y.refresh()


    ok1_tuple=(1,k,o)
    
    for i in range(len(_perm2d)):
        tmpTrans = N.transpose(a, _perm2d[i]).copy() # force contiguous!
        aWV.shape = tmpTrans.shape

        ok1 = tuple( map(int, N.take(ok1_tuple, _perm2d[i])[::-1] ) )
        apply(W.fastwv6, (tmpTrans, aWV) + ok1 + (verboseLvl>=2,))

        tmpTrans = N.transpose(aWV, _invperm2d[i])
        tmpTrans /= len(_perm2d) * nAngles
        res += tmpTrans

    # now do it for a bunch of angles
    # angles = range(30,90,  30) # won't include 90 ...

    angles = N.arange(90./nAngles,  90., 90./nAngles)

    for angle in angles:
        if verboseLvl >=1:
            print "begin angle ",angle
            Y.refresh()

        U.rot3d(a,aRot, angle, 0)
        c[:]=0

        for i in range(len(_perm2d)):

            tmpTrans = N.transpose(aRot, _perm2d[i]).copy() # force contiguous!
            aWV.shape = tmpTrans.shape

            ok1 = tuple( map(int, N.take(ok1_tuple, _perm2d[i])[::-1] ) )
            apply(W.fastwv6, (tmpTrans, aWV) + ok1 + (verboseLvl>=2,))

            tmpTrans  = N.transpose(aWV, _invperm2d[i])
            tmpTrans /= len(_perm2d) * nAngles
            c += tmpTrans

        U.rot3d(c,aRot, -angle, 0)
        res += aRot



def twoDt(a, res, sXY,sT, nAngles=6):

    perm2dt = (
    (0,1,2),
    (0,2,1),
    (2,1,0),
    (1,2,0),
    )
    invperm2dt = (
    (0,1,2),
    (0,2,1),
    (2,1,0),
    (2,0,1),
    )

    if sXY < 1 or sT < 1:
        raise ValueError, "order need to >= 1"
    if nAngles < 1:
        raise ValueError, "nAngles need to >= 1"
    res[:] = 0

    okt_tuple=(sT,sXY,sXY)
    d = F.zeroArrF(a.shape)
    #seb  r = F.zeroArrF(a.shape)
    s = F.zeroArrF(a.shape)

    # do first wave3dChp on original data

    for i in range(len(perm2dt)):
        #print "perm=",perm2dt[i]
        #print "invperm=",invperm2dt[i]
        b = N.transpose(a, perm2dt[i]).copy() # force contiguous!
        c = F.zeroArrF(b.shape)
        # awkward calling sequence to insure loading of proper indices
        okt = tuple( map(int, N.take(okt_tuple, perm2dt[i])[::-1] ) )
        #print "tuple:", okt
        #print "  "
        apply(W.fastwv6,(b,c)+okt)
        e = N.transpose(c,invperm2dt[i])
        d += e

    d /= len(perm2dt)
    res += d

### P.save(d, "sym"+fn + "_" + str(o) + "_rot0.mrc")


    # now do it for a bunch of angles

    aa = F.zeroArrF(a.shape)

    # angles = range(30,90,  30) # won't include 90 ...

    angles = N.arange(90./nAngles,  90., 90./nAngles)

    for angle in angles:
        print "begin angle ",angle
        Y.refresh()
        axis = 0
        d[:,:,:]=0
        U.rot3d(a,aa, angle, axis)

        for i in range(len(perm2dt)):
            #print "perm=",perm2dt[i]
            #print "invperm=",invperm2dt[i]
            b = N.transpose(aa, perm2dt[i]).copy() # force contiguous!
            c = F.zeroArrF(b.shape)
            okt = tuple( map(int, N.take(okt_tuple, perm2dt[i])[::-1] ) )
            #print "tuple:", okt
            #print " "
            apply(W.fastwv6,(b,c)+ okt)
            e = N.transpose(c,invperm2dt[i])
            d += e

        d /= len(perm2dt)
        U.rot3d(d,s,-angle,axis)
        res += s


    res /= (1+len(angles))


    #seb  # if oz == o:
    #seb    if int(nAngles) != 2:
    #seb        P.save( r/ (1+len(angles)), fn + "_wvl2Dt_" + str(o) + "_"+str(k)+"_"+str(l)+"_"+str(nAngles)+"angles")
    #seb    else:
    #seb        P.save( r/ (1+len(angles)), fn + "_wvl2Dt_" + str(o) + "_"+str(k)+"_"+str(l))






def twoDF(a,res, o,k, ndegrees):

    loop_tup = ((1,k,o),(1,o,k))
    #seb    a = P.load(hd+fn + ".mrc")
    #u = F.zeroArrF(a.shape)

    #ok_tuple = (1, k,o) # wavelet tuple in z,y,x order
    ok_tuple = loop_tup[0] # wavelet tuple in z,y,x order
    ###a = P.load("../"+fn + ".mrc")
    d = F.zeroArrF(a.shape)
    #seb r = F.zeroArrF(a.shape)
    aa = F.zeroArrF(a.shape)

    #seb wx.wxYield()
    res[:]=0  #seb

    #d[:,:,:]=0
    U.rot3d(a,aa, ndegrees, 0)
    for i in range(len(perm2d)):
        print perm2d[i],
        #print invperm2d[i]
        b = N.transpose(aa, perm2d[i]).copy() # force contiguous!
        c = F.zeroArrF(b.shape)
        okt = tuple( map(int, N.take(ok_tuple, perm2d[i])[::-1] ) )
        print "tuple:", okt
        apply(W.wave3dChp,  (b,c) + okt)
        #old:  W.wave3dChp(b,c, o,k,1)
        e = N.transpose(c,invperm2d[i])
        d += e

    d /= len(perm2d)
    U.rot3d(d,res,-ndegrees,0)
    #seb    P.save( r, fn + "_" + str(o) + "_" + str(k)+ "_"+str(ndegrees)+"_degrees"+".mrc")








def oneDT(a, res, o,l):

    perm1dt = ((0,1,2),(0,2,1))
    invperm1dt = perm1dt


    okt_tuple=(1,l,o)
    #seb a = P.load(hd+fn)
    #seb d = F.zeroArrF(a.shape)
    #seb r = F.zeroArrF(a.shape)
    #seb s = F.zeroArrF(a.shape)
    #seb nAngles=1

    res[:] = 0 #seb 

    # do first wave3dChp on original data

    for i in range(len(perm1dt)):
        b = N.transpose(a, perm1dt[i]).copy() # force contiguous!
        #11111111111111c.shape = b.shape
        c = F.zeroArrF(a.shape)
        # awkward calling sequence to insure loading of proper indices
        okt = tuple( map(int, N.take(okt_tuple, perm1dt[i])[::-1] ) )
        #print "tuple:", okt
        #print "  "
        apply(W.fastwv6,(b,c)+okt)
        e = N.transpose(c,invperm1dt[i])
        res += e

    res /= len(perm1dt)
    #seb r += d

### P.save(d, "sym"+fn + "_" + str(o) + "_rot0.mrc")



    #seb    aa = F.zeroArrF(a.shape)


    #seb angles = N.arange(90./nAngles,  90., 90./nAngles)

    #seb    for angle in angles:
    #seb        #print "begin angle ",angle
    #seb        print "if you can read this, there is an error"
    #seb        wx.wxYield()
    #seb        axis = 0
    #seb        d[:,:,:]=0
    #seb        U.rot3d(a,aa, angle, axis)
    #seb 
    #seb        for i in range(len(perm1dt)):
    #seb            b = N.transpose(aa, perm1dt[i])
    #seb            c = F.zeroArrF(b.shape)
    #seb            okt = tuple( N.take(okt_tuple, perm1dt[i])[::-1] )
    #seb            #print "tuple:", okt
    #seb            #print " "
    #seb            apply(Wvl.W.fastwv6,(b,c)+ okt)
    #seb            e = N.transpose(c,invperm1dt[i])
    #seb            d += e
    
    #seb    d /= len(perm1dt)
    #seb    U.rot3d(d,s,-angle,axis)
    #seb    r += s

        

    #seb P.save( r/ (1+len(angles)), fn + "_wvl1Dt_" + str(o) + "_"+str(l))
