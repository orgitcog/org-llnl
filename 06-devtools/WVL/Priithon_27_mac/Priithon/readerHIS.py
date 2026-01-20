"""
Priithon: Hamamatsu Image Sequence file format reader
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import numpy as N

mmap_shape = None # default: map entire file; change this to handle BIG files

# 64 bytes
dtypeHIS = N.dtype([
    ('magic', 'a2'),
    ('ComLen', N.uint16),
    ('iDX', N.uint16),
    ('iDY', N.uint16),
    ('iX', N.uint16),
    ('iY', N.uint16),
    ('pixType', N.uint16),
    ('numImgs', N.uint32),
    ('numChan', N.uint16),
    ('chan', N.uint16),
    ('timeStamp', N.float64),
    ('marker', N.uint32),
    ('miscinfo', '30i1'),
    ])

# pixType
hisType2numpyDtype = {
    1: N.uint8,
    2: N.uint16,
    3: N.uint32,
    11: ('RGB', (N.uint8,N.uint8,N.uint8)),
    12: ('RGB', (N.uint16,N.uint16,N.uint16)),
}

#20100224 class ndarray_inHisFile(N.ndarray):
#20100224     def __array_finalize__(self,obj):
#20100224         self.HIS = getattr(obj, 'HIS', None)

#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
# Simple example - adding an extra attribute to ndarray
class ndarray_inHisFile(N.ndarray):
    def __new__(cls, input_array, hisInfo=None):
        obj = N.asanyarray(input_array).view(cls)
        obj.HIS = hisInfo
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.HIS = getattr(obj, 'HIS', None)
        self.__dict__.update(getattr(obj, "__dict__", {}))

def _try_openHIS_fastMap(m):
    hisHdr0 = m[:64]
    hisHdr0.dtype = dtypeHIS
    try:
        hisHdr0 = hisHdr0[0]
    except IndexError:
        raise EOFError, "zero Bytes HIS file"

    imgPixDType = hisType2numpyDtype[ hisHdr0['pixType'] ]
    pixBytes = imgPixDType().itemsize
    nx,ny,nz = hisHdr0['iDX'],  hisHdr0['iDY'],  hisHdr0['numImgs']
    comLen=hisHdr0['ComLen']

    expectedBytes = (64 + pixBytes*nx*ny) * nz + comLen
    if expectedBytes != len(m):
        return None # there are probably comments in other sections, fastMap cannot be used

    mm = m[comLen:] # first hdr will be "corrupt", since comment is just before first imgData
    a = N.recarray(nz, dtype=[( 'hdr',     dtypeHIS ), 
                              ( 'imgData', (imgPixDType, (ny,nx)) ),
                              ], 
                   buf=mm)
    if comLen:
        hisComment = m[64:64+comLen]
        hisComment.dtype = '|S%d'%(comLen,)
    else:
        hisComment = ('',)
    comment = hisComment[0]  # there is "one" comment per sect

    class hisInfo:
        hdr0 = hisHdr0
        comment0 = comment
        hdr = a['hdr']

    fastHisArr = ndarray_inHisFile(a['imgData'], hisInfo=hisInfo)
    return fastHisArr

def openHIS(fn, mode='r'):
    """
    open Hamamatsu Image Sequence
    return 3d array of all sections
    if only the first section has a non-zero length variable comment entry
       (or no sections have a comment) 
       "fast mapping" should work and the whole file can be memmap'ed into one
       3D memmap array, with a special `HIS` attribute containing
          hdr0 = hisHdr0      # HIS-header info of first section
          comment0 = comment  # comment of first section
          hdr = a['hdr']      # HIS-header info all sections
               # NOTE:  `hdr[0]` IS 'CORRUPT' !!! DON'T USE THIS 1
                        USE `hdr0` INSTEAD !!!! 
    otherwise
       this returns a mockNDarray of memmaped sections
       each section contains a HIS attribute,
          which contains hdr, offsetNext and comment
    """

    m = N.memmap(fn, shape=mmap_shape, mode=mode)

    if mmap_shape is None:
        a = _try_openHIS_fastMap(m)
        if a is not None:
            return a

    nExpected = None

    offset=0
    imgs = []
    while 1: # for i in range(10):
        try:
            img = readSection(m, offset)
        except EOFError:#  EOFError:
            break
        except ValueError: 
            # img.shape = hisHdr['iDY'], hisHdr['iDX']
            #ValueError: total size of new array must be unchanged
            if nExpected is not None:
                if nExpected == len(imgs):
                    print " *** found extra trailing bytes in HIS file -- ignoring since number of frames as expected from header: %d ***" %(nExpected,)
                else:
                    print " *** found extraneous trailing bytes in HIS file -- ignoring, BUT found number of frames  <> expected from header: %d != %d ***" %(
                     len(imgs), nExpected)
                break
        else:
            if nExpected is None:
                nExpected = img.HIS.hdr['numImgs']
        imgs.append(img)
        offset = img.HIS.offsetNext

    from .fftfuncs import mockNDarray
    return mockNDarray(*imgs)


def readSection(m, offsetSect = 0):
    """
    m:          numpy memmap of a file
    offsetSect: offset of first byte of section to be read
    """

    offsetComment = offsetSect + 64

    hisHdr = m[offsetSect:offsetComment]
    hisHdr.dtype = dtypeHIS
    try:
        hisHdr = hisHdr[0]
    except IndexError:
        raise EOFError, "End of HIS file reached"


    assert hisHdr['magic'] == 'IM'

    commentLength = hisHdr['ComLen']
    offsetImg     = offsetComment + commentLength

    if commentLength:
        hisComment = m[offsetComment:offsetImg]
        hisComment.dtype = '|S%d'%(hisHdr['ComLen'],)
    else:
        hisComment = ('',)
    imgPixDType = hisType2numpyDtype[ hisHdr['pixType'] ]
    imgBytes = int(hisHdr['iDX']) * int(hisHdr['iDY']) * imgPixDType().itemsize
    
    sectEnd = offsetImg + imgBytes

    img = m[offsetImg:sectEnd]
    img.dtype = imgPixDType
    img.shape = hisHdr['iDY'], hisHdr['iDX']


    #import weakref
    #hisHdr     = weakref.proxy( hisHdr )
    #hisComment = weakref.proxy( hishisComment )
    
#20100224     img.__class__ = ndarray_inHisFile
    class hisHeaderInfo:
        hdr = hisHdr
        comment = hisComment[0]  # there is "one" comment per sect
        offsetNext = sectEnd

#20100224     img.HIS = hisHeaderInfo
    img = ndarray_inHisFile(img, hisInfo=hisHeaderInfo)

    return img

'''
def loadHISsec0(fn, sec=0):
    #import os
    f = open(fn, "rb")
    offsetSect = 0
    iSec = 0
    global sectsOffsets, hisHdr
    sectsOffsets = []
    
    while 1:
        hisHdr = N.fromfile(f, dtype=dtypeHIS, count=1)
        hisHdr = hisHdr[0]

        assert hisHdr['magic'] == 'IM'
        offsetComment = offsetSect + 64
        commentLength = hisHdr['ComLen']
        offsetImg     = offsetComment + commentLength
        imgPixDType = hisType2numpyDtype[ hisHdr['pixType'] ]
        imgBytes = int(hisHdr['iDX']) * int(hisHdr['iDY']) * imgPixDType().itemsize
        #imgShape = hisHdr['iDY'], hisHdr['iDX']

        sectEnd = offsetImg + imgBytes
        offsetSect = sectEnd
        sectsOffsets.append( offsetSect )
        f.seek( offsetSect )
        #f.seek(offsetSect+64+commentLength+imgBytes)  os.SEEK_CUR)
'''

def loadHISsec(fn, sec=0):
    #import os
    f = open(fn, "rb")
    offsetSect = 0
    iSec = 0
    global sectsOffsets, hisHdr, hisComment
    #sectsOffsets = []
    
    for i in range(sec+1):
        f.seek( offsetSect )
        hisHdr = N.fromfile(f, dtype=dtypeHIS, count=1)
        hisHdr = hisHdr[0]

        assert hisHdr['magic'] == 'IM'
        offsetComment = offsetSect + 64
        commentLength = hisHdr['ComLen']
        offsetImg     = offsetComment + commentLength
        imgPixDType = hisType2numpyDtype[ hisHdr['pixType'] ]
        imgBytes = int(hisHdr['iDX']) * int(hisHdr['iDY']) * imgPixDType().itemsize
        
        sectEnd = offsetImg + imgBytes
        offsetSect = sectEnd
        #sectsOffsets.append( offsetSect )
        #f.seek(offsetSect+64+commentLength+imgBytes)  os.SEEK_CUR)

    
    hisComment = f.read(commentLength)
    imgShape = hisHdr['iDY'], hisHdr['iDX']
    img = N.fromfile(f, dtype=imgPixDType, count=N.prod(imgShape))
    img.shape = imgShape = hisHdr['iDY'], hisHdr['iDX']

    #img.__class__ = ndarray_inHisFile
    #img = ndarray_inHisFile(img)
    #class hisHeaderInfo:
    #    hdr = hisHdr
    #    comment = hisComment
    #    offsetNext = sectEnd

    #img.HIS = hisHeaderInfo
    return img

def loadHISsects(fn, secStart=0, secEnd=1, stride=1):
    #import os
    f = open(fn, "rb")
    offsetSect = 0
    iSec = 0
    imgs = []
    #sectsOffsets = []
    
    i=0
    while secEnd is None or i < secEnd:
    #for i in range(secEnd):
        f.seek( offsetSect )
        hisHdr = N.fromfile(f, dtype=dtypeHIS, count=1)
        hisHdr = hisHdr[0]

        assert hisHdr['magic'] == 'IM'
        offsetComment = offsetSect + 64
        commentLength = hisHdr['ComLen']
        offsetImg     = offsetComment + commentLength
        imgPixDType = hisType2numpyDtype[ hisHdr['pixType'] ]
        imgBytes = int(hisHdr['iDX']) * int(hisHdr['iDY']) * imgPixDType().itemsize
        
        sectEnd = offsetImg + imgBytes
        offsetSect = sectEnd
        #sectsOffsets.append( offsetSect )
        #f.seek(offsetSect+64+commentLength+imgBytes)  os.SEEK_CUR)

        if i >= secStart and (((i - secStart)%stride)==0):
            hisComment = f.read(commentLength)
            imgShape = hisHdr['iDY'], hisHdr['iDX']
            img = N.fromfile(f, dtype=imgPixDType, count=N.prod(imgShape))
            img.shape = imgShape = hisHdr['iDY'], hisHdr['iDX']

            #img.__class__ = ndarray_inHisFile
            #img = ndarray_inHisFile(img)
            #class hisHeaderInfo:
            #    hdr = hisHdr
            #    comment = hisComment
            #    offsetNext = sectEnd
            
            imgs.append( img )
        i+=1
    #img.HIS = hisHeaderInfo

    from .fftfuncs import mockNDarray
    return mockNDarray(*imgs)
