# read_spe.py

"""
Priithon: reader for Hamamatsu(?) SPE files

# http://www.scipy.org/Cookbook/Reading_SPE_files
# http://forums.wolfram.com/student-support/topics/8778

# MORE INFO AT http://ftp.roperscientific.com/Public/Manuals/Princeton%20Instruments/
#     WinSpec32.pdf
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import numpy as N


# variable here only for documentation purposes
_header_struct__info = """ 
typedef struct tagCSMAHEAD {
#ifdef __cplusplus                // STARTING
  tagCSMAHEAD();                  // ADDRESS
#endif                            // (decimal)
  SHORT   ControllerVersion;      //    0  Hardware Version
  SHORT   LogicOutput;            //    2  Definition of Output BNC
  WORD    AmpHiCapLowNoise;       //    4  Amp Switching Mode
  WORD    xDimDet;                //    6  Detector x dimension of chip.
  SHORT   mode;                   //    8  timing mode
  float   exp_sec;                //   10  alternitive exposure, in sec.
  SHORT   VChipXdim;              //   14  Virtual Chip X dim
  SHORT   VChipYdim;              //   16  Virtual Chip Y dim
  WORD    yDimDet;                //   18  y dimension of CCD or detector.
  char    date[DATEMAX];          //   20  date
  SHORT   VirtualChipFlag;        //   30  On/Off
  char    Spare_1[2];             //   32
  SHORT   noscan;                 //   34  Old number of scans - should always be -1
  float   DetTemperature;         //   36  Detector Temperature Set
  SHORT   DetType;                //   40  CCD/DiodeArray type
  WORD    xdim;                   //   42  actual # of pixels on x axis
  SHORT   stdiode;                //   44  trigger diode
  float   DelayTime;              //   46  Used with Async Mode
  WORD    ShutterControl;         //   50  Normal, Disabled Open, Disabled Closed
  SHORT   AbsorbLive;             //   52  On/Off
  WORD    AbsorbMode;             //   54  Reference Strip or File
  SHORT   CanDoVirtualChipFlag;   //   56  T/F Cont/Chip able to do Virtual Chip
  SHORT   ThresholdMinLive;       //   58  On/Off
  float   ThresholdMinVal;        //   60  Threshold Minimum Value
  SHORT   ThresholdMaxLive;       //   64  On/Off
  float   ThresholdMaxVal;        //   66  Threshold Maximum Value
  SHORT   SpecAutoSpectroMode;    //   70  T/F Spectrograph Used
  float   SpecCenterWlNm;         //   72  Center Wavelength in Nm
  SHORT   SpecGlueFlag;           //   76  T/F File is Glued
  float   SpecGlueStartWlNm;      //   78  Starting Wavelength in Nm
  float   SpecGlueEndWlNm;        //   82  Starting Wavelength in Nm
  float   SpecGlueMinOvrlpNm;     //   86  Minimum Overlap in Nm
  float   SpecGlueFinalResNm;     //   90  Final Resolution in Nm
  SHORT   PulserType;             //   94  0=None, PG200=1, PTG=2, DG535=3
  SHORT   CustomChipFlag;         //   96  T/F Custom Chip Used
  SHORT   XPrePixels;             //   98  Pre Pixels in X direction
  SHORT   XPostPixels;            //  100  Post Pixels in X direction
  SHORT   YPrePixels;             //  102  Pre Pixels in Y direction 
  SHORT   YPostPixels;            //  104  Post Pixels in Y direction
  SHORT   asynen;                 //  106  asynchron enable flag  0 = off
  SHORT   datatype;               //  108  experiment datatype
                                  //       0 =   FLOATING POINT
                                  //       1 =   LONG INTEGER
                                  //       2 =   INTEGER
                                  //       3 =   UNSIGNED INTEGER
  SHORT   PulserMode;             //  110  Repetitive/Sequential
  USHORT  PulserOnChipAccums;     //  112  Num PTG On-Chip Accums
  DWORD   PulserRepeatExp;        //  114  Num Exp Repeats (Pulser SW Accum)
  float   PulseRepWidth;          //  118  Width Value for Repetitive pulse (usec)
  float   PulseRepDelay;          //  122  Width Value for Repetitive pulse (usec)
  float   PulseSeqStartWidth;     //  126  Start Width for Sequential pulse (usec)
  float   PulseSeqEndWidth;       //  130  End Width for Sequential pulse (usec)
  float   PulseSeqStartDelay;     //  134  Start Delay for Sequential pulse (usec)
  float   PulseSeqEndDelay;       //  138  End Delay for Sequential pulse (usec)
  SHORT   PulseSeqIncMode;        //  142  Increments: 1=Fixed, 2=Exponential
  SHORT   PImaxUsed;              //  144  PI-Max type controller flag
  SHORT   PImaxMode;              //  146  PI-Max mode
  SHORT   PImaxGain;              //  148  PI-Max Gain
  SHORT   BackGrndApplied;        //  150  1 if background subtraction done
  SHORT   PImax2nsBrdUsed;        //  152  T/F PI-Max 2ns Board Used
  WORD    minblk;                 //  154  min. # of strips per skips
  WORD    numminblk;              //  156  # of min-blocks before geo skps
  SHORT   SpecMirrorLocation[2];  //  158  Spectro Mirror Location, 0=Not Present
  SHORT   SpecSlitLocation[4];    //  162  Spectro Slit Location, 0=Not Present
  SHORT   CustomTimingFlag;       //  170  T/F Custom Timing Used
  char    ExperimentTimeLocal[TIMEMAX];//  172  Experiment Local Time as hhmmss\0
  char    ExperimentTimeUTC[TIMEMAX];  //  179  Experiment UTC Time as hhmmss\0
  SHORT   ExposUnits;             //  186  User Units for Exposure
  WORD    ADCoffset;              //  188  ADC offset
  WORD    ADCrate;                //  190  ADC rate
  WORD    ADCtype;                //  192  ADC type
  WORD    ADCresolution;          //  194  ADC resolution
  WORD    ADCbitAdjust;           //  196  ADC bit adjust
  WORD    gain;                   //  198  gain
  char    Comments[5][COMMENTMAX];//  200  File Comments
  WORD    geometric;              //  600  geometric ops: rotate 0x01,
                                  //       reverse 0x02, flip 0x04
  char    xlabel[LABELMAX];       //  602  intensity display string
  WORD    cleans;                 //  618  cleans
  WORD    NumSkpPerCln;           //  620  number of skips per clean.
  SHORT   SpecMirrorPos[2];       //  622  Spectrograph Mirror Positions
  float   SpecSlitPos[4];         //  626  Spectrograph Slit Positions
  SHORT   AutoCleansActive;       //  642  T/F
  SHORT   UseContCleansInst;      //  644  T/F
  SHORT   AbsorbStripNum;         //  646  Absorbance Strip Number
  SHORT   SpecSlitPosUnits;       //  648  Spectrograph Slit Position Units
  float   SpecGrooves;            //  650  Spectrograph Grating Grooves
  SHORT   srccmp;                 //  654  number of source comp. diodes
  WORD    ydim;                   //  656  y dimension of raw data.
  SHORT   scramble;               //  658  0=scrambled,1=unscrambled
  SHORT   ContinuousCleansFlag;   //  660  T/F Continuous Cleans Timing Option
  SHORT   ExternalTriggerFlag;    //  662  T/F External Trigger Timing Option
  long    lnoscan;                //  664  Number of scans (Early WinX)
  long    lavgexp;                //  668  Number of Accumulations
  float   ReadoutTime;            //  672  Experiment readout time
  SHORT   TriggeredModeFlag;      //  676  T/F Triggered Timing Option
  char    Spare_2[10];            //  678  
  char    sw_version[FILEVERMAX]; //  688  Version of SW creating this file
  SHORT   type;                   //  704  0=1000,1=new120,2=old120,3=130,
                                  //       st121=4,st138=5,dc131(PentaMax)=6,
                                  //       st133(MicroMax)=7,st135(GPIB)=8,
                                  //       VICCD=9, ST116(GPIB)=10,
                                  //       OMA3(GPIB)=11,OMA4=12
  SHORT   flatFieldApplied;       //  706  1 if flat field was applied.
  char    Spare_3[16];            //  708  
  SHORT   kin_trig_mode;          //  724  Kinetics Trigger Mode
  char    dlabel[LABELMAX];       //  726  Data label.
  char    Spare_4[436];           //  742
  char    PulseFileName[HDRNAMEMAX]; //  1178  Name of Pulser File with
                                  //           Pulse Widths/Delays (for Z-Slice)
  char    AbsorbFileName[HDRNAMEMAX]; // 1298 Name of Absorbance File (if File Mode)
  DWORD   NumExpRepeats;          // 1418  Number of Times experiment repeated
  DWORD   NumExpAccums;           // 1422  Number of Time experiment accumulated
  SHORT   YT_Flag;                // 1426  Set to 1 if this file contains YT data
  float   clkspd_us;              // 1428  Vert Clock Speed in micro-sec
  SHORT   HWaccumFlag;            // 1432  set to 1 if accum done by Hardware.
  SHORT   StoreSync;              // 1434  set to 1 if store sync used.
  SHORT   BlemishApplied;         // 1436  set to 1 if blemish removal applied.
  SHORT   CosmicApplied;          // 1438  set to 1 if cosmic ray removal applied
  SHORT   CosmicType;             // 1440  if cosmic ray applied, this is type.
  float   CosmicThreshold;        // 1442  Threshold of cosmic ray removal.
  long    NumFrames;              // 1446  number of frames in file.
  float   MaxIntensity;           // 1450  max intensity of data (future)
  float   MinIntensity;           // 1454  min intensity of data (future)
  char    ylabel[LABELMAX];       // 1458  y axis label.
  WORD    ShutterType;            // 1474  shutter type.
  float   shutterComp;            // 1476  shutter compensation time.
  WORD    readoutMode;            // 1480  readout mode, full,kinetics, etc
  WORD    WindowSize;             // 1482  window size for kinetics only.
  WORD    clkspd;                 // 1484  clock speed for kinetics & frame transfer.
  WORD    interface_type;         // 1486  computer interface
                                  //       (isa-taxi, pci, eisa, etc.)
  SHORT   NumROIsInExperiment;    // 1488  May be more than the 10 allowed in
                                  //       this header (if 0, assume 1)
  char    Spare_5[16];            // 1490
  WORD    controllerNum;          // 1506  if multiple controller system will
                                  //       have controller number data came from.
                                  //       this is a future item.
  WORD    SWmade;                 // 1508  Which software package created this file
  SHORT   NumROI;                 // 1510  number of ROIs used. if 0 assume 1.
  ROIinfo ROIinfoblk[ROIMAX];     // 1512  ROI information
  char    FlatField[HDRNAMEMAX];  // 1632  Flat field file name.
  char    background[HDRNAMEMAX]; // 1752  background sub. file name.
  char    blemish[HDRNAMEMAX];    // 1872  blemish file name.
  float   file_header_ver;        // 1992  version of this file header
  char    YT_Info[1000];          // 1996-2996  Reserved for YT information
  LONG    WinView_id;             // 2996  == 0x01234567L if in use by WinView
  struct {                        //  X     Y
    double offset;                // 3000  3489  offset for absolut data scaling
    double factor;                // 3008  3497  factor for absolut data scaling
    char   current_unit;          // 3016  3505  selected scaling unit
    char   CalibReserved1;        // 3017  3506  reserved
    char   display_string[40];    // 3018  3507  string used for drawing axes...
    char   CalibReserved2[40];    // 3058  3547  reserved
    char   calib_valid;           // 3098  3587  flag if calibration is valid
    char   input_unit;            // 3099  3588  current input units for "calib_value"
    char   polynounit;            // 3100  3589  linear UNIT and used in the "polynocoeff"
    char   polynoorder;           // 3101  3590  ORDER of calibration POLYNOM
    char   calib_count;           // 3102  3591  valid calibration data pairs
    double pixel_position[10];    // 3103  3592  pixel pos. of calibration data
    double calib_value[10];       // 3183  3672  calibration VALUE at above pos
    double polynocoeff[6];        // 3263  3752  polynom COEFFICIENTS
    double laser_position;        // 3311  3800  laser wavenumber for relativ WN
    char   CalibReserved3;        // 3319  3808  reserved
    BYTE   leftover_flag;         // 3320  3809  not used.
    char   user_label[40];        // 3321  3810  Calibration label
    char   expansion[128];        // 3361  3850  Expansion area
  } xcalibration,                 // 3000 -> 3488  X axis calibration
    ycalibration;                 // 3489 -> 3977  Y axis calibration
  char    Istring[40];            // 3978  special intensity scaling string
  char    Spare_6[80];            // 4018  
  SHORT   lastvalue;              // 4098  Always the LAST value in the header
} CSMAHead;                       // 4100 Bytes Total Header Size
"""



_type_dtype_map = { 
    0: N.float32,
    1: N.int32,
    2: N.int16,
    3: N.uint16,
}
class File(object):
    
    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = N.int64(self.read_at(42, 1, N.uint16)[0])
        self._ydim = N.int64(self.read_at(656, 1, N.uint16)[0])
        self._zdim = N.int64(self.read_at(1446, 1, N.int32)[0])
        self._type = self.read_at(108, 1, N.int16)[0]
        self._dtype = _type_dtype_map[ self._type ]


    def get_shape(self):
        return (self._zdim, self._ydim, self._xdim)

    def read_at(self, pos, size, ntype):
        #use memmap instead  self._fid.seek(pos)
        #use memmap instead  return N.fromfile(self._fid, ntype, size)
        return N.memmap(self._fid, dtype=ntype, mode='r', offset=pos, shape=size)

    def load_img(self):
        img = self.read_at(4100, self._zdim * self._xdim * self._ydim, self._dtype)
        img.shape = self.get_shape()
        return img

    def close(self):
        self._fid.close()

def load(fname):
    fid = File(fname)
    img = fid.load_img()
    fid.close()
    return img

if __name__ == "__main__":
    import sys
    img = load(sys.argv[-1])


###############################################################################
###############################################################################
##
##  http://ftp.piacton.com/public/software/User%20Written%20Utilities/python/piUtils.py
##
###############################################################################
###############################################################################

# piUtils.py module "Princeton Instruments Utilities"
# 
# Process SPE data generated by Princeton Instruments cameras
# Convert SPE images into the more commonly used FITS format.
#
# A typical user will likely only call speToFits() (rather than
# readSpe() and speDictToFits()).  speToFits() takes an SPE
# filename and a base name for the FITS files, and produces a
# sequence of FITS files, one per image frame in the SPE file.
#
# I wrote this code originally to handle SPE data from
# the PI-Max ICCD camera.  I have not tested this code
# on SPE data from other PI cameras.
#
# If you improve/extend the code, I'd love to hear about it.
# Please contact me at:
#    James Battat
#    jbattat@post.harvard.edu
#
# 2009 April
#

import sys
import os.path
import struct # deal with binary data
import numpy
import pyfits

def speToFits(spefile, fitsfile=None, clobber=False, verbose=False):
  """ Read in an SPE file and output a FITS file

  Example usage:
  Given an SPE file named test.SPE which contains 3 image frames,
  then the following code will produce three FITS images, named:
  .   image_x001.fits
  .   image_x002.fits
  .   image_x003.fits
  >>> import piUtils
  >>> piUtils.speToFits('test.SPE', fitsfile='image')
  """
  spedict = readSpe(spefile)
  speDictToFits(spedict, outfile=fitsfile, clobber=clobber, verbose=verbose)
  return

def readSpe(spefilename, verbose=False, onlyHdr=False):
  """ Read a binary PI SPE file into a python dictionary

  Inputs
  spefilename --  string specifying the name of the SPE file to be read
  verbose     --  boolean print debug statements (True) or not (False)
  onlyHdr     --  if True, only read header, and skip filling `data` with image data

  Outputs
  spedict     --  python dictionary containing header and data information
                  from the SPE file
                  Content of the dictionary is:
                  spedict = {'data':[],    # a list of 2D numpy arrays, one per image
                  'IGAIN':pimaxGain,
                  'EXPOSURE':exp_sec,
                  'SPEFNAME':spefilename,
                  'OBSDATE':date,
                  'CHIPTEMP':detectorTemperature
             }

  Example usage:
  Given an SPE file named test.SPE, you can read the SPE data into
  a python dictionary named spedict with the following:
  >>> import piUtils
  >>> spedict = piUtils.readSpe('test.SPE')
  """
  
  # open SPE file as binary input
  spe = open(spefilename, "rb")
  
  # Header length is a fixed number
  nBytesInHeader = 4100

  # Read the entire header
  header = spe.read(nBytesInHeader)
  
  # version of WinView used
  swversion = struct.unpack_from("16s", header, offset=688)[0]
  
  # version of header used
  # Eventually, need to adjust the header unpacking
  # based on the headerVersion.  
  headerVersion = struct.unpack_from("f", header, offset=1992)[0]
  
  # which camera controller was used?
  controllerVersion = struct.unpack_from("h", header, offset=0)[0]
  if verbose:
    print "swversion         = ", swversion
    print "headerVersion     = ", headerVersion
    print "controllerVersion = ", controllerVersion
  
  # Date of the observation
  # (format is DDMONYYYY  e.g. 27Jan2009)
  date = struct.unpack_from("9s", header, offset=20)[0]
  
  # Exposure time (float)
  exp_sec = struct.unpack_from("f", header, offset=10)[0]
  
  # Intensifier gain
  pimaxGain = struct.unpack_from("h", header, offset=148)[0]

  # Not sure which "gain" this is
  gain = struct.unpack_from("H", header, offset=198)[0]
  
  # Data type (0=float, 1=long integer, 2=integer, 3=unsigned int)
  data_type = struct.unpack_from("h", header, offset=108)[0]

  comments = struct.unpack_from("400s", header, offset=200)[0]

  # CCD Chip Temperature (Degrees C)
  detectorTemperature = struct.unpack_from("f", header, offset=36)[0]

  # The following get read but are not used
  # (this part is only lightly tested...)
  analogGain = struct.unpack_from("h", header, offset=4092)[0]
  noscan = struct.unpack_from("h", header, offset=34)[0]
  pimaxUsed = struct.unpack_from("h", header, offset=144)[0]
  pimaxMode = struct.unpack_from("h", header, offset=146)[0]

  if verbose:
    print "date      = ["+date+"]"
    print "exp_sec   = ", exp_sec
    print "pimaxGain = ", pimaxGain
    print "gain (?)  = ", gain
    print "data_type = ", data_type
    print "comments  = ["+comments+"]"
    print "analogGain = ", analogGain
    print "noscan = ", noscan
    print "detectorTemperature [C] = ", detectorTemperature
    print "pimaxUsed = ", pimaxUsed

  # Determine the data type format string for
  # upcoming struct.unpack_from() calls
  if data_type == 0:
    # float (4 bytes)
    dataTypeStr = "f"  #untested
    bytesPerPixel = 4
    dtype = "float32"
  elif data_type == 1:
    # long (4 bytes)
    dataTypeStr = "l"  #untested
    bytesPerPixel = 4
    dtype = "int32"
  elif data_type == 2:
    # short (2 bytes)
    dataTypeStr = "h"  #untested
    bytesPerPixel = 2
    dtype = "int32"
  elif data_type == 3:  
    # unsigned short (2 bytes)
    dataTypeStr = "H"  # 16 bits in python on intel mac
    bytesPerPixel = 2
    dtype = "int32"  # for numpy.array().
    # other options include:
    # IntN, UintN, where N = 8,16,32 or 64
    # and Float32, Float64, Complex64, Complex128
    # but need to verify that pyfits._ImageBaseHDU.ImgCode cna handle it
    # right now, ImgCode must be float32, float64, int16, int32, int64 or uint8
  else:
    print "unknown data type"
    print "returning..."
    sys.exit()
  
  # Number of pixels on x-axis and y-axis
  nx = struct.unpack_from("H", header, offset=42)[0]
  ny = struct.unpack_from("H", header, offset=656)[0]
  
  # Number of image frames in this SPE file
  nframes = struct.unpack_from("l", header, offset=1446)[0]

  if verbose:
    print "nx, ny, nframes = ", nx, ", ", ny, ", ", nframes
  
  npixels = nx*ny
  npixStr = str(npixels)
  fmtStr  = npixStr+dataTypeStr
  if verbose:
    print "fmtStr = ", fmtStr
  
  # How many bytes per image?
  nbytesPerFrame = npixels*bytesPerPixel
  if verbose:
    print "nbytesPerFrame = ", nbytesPerFrame

  # Create a dictionary that holds some header information
  # and contains a placeholder for the image data
  spedict = {'data':[],    # can have more than one image frame per SPE file
             'IGAIN':pimaxGain,
             'EXPOSURE':exp_sec,
             'SPEFNAME':spefilename,
             'OBSDATE':date,
             'CHIPTEMP':detectorTemperature
             }
  
  # Now read in the image data
  # Loop over each image frame in the image
  if verbose:
    print "Reading image frames number ",

  if not onlyHdr:
      for ii in range(nframes):
          iistr = str(ii)
          data = spe.read(nbytesPerFrame)
          if verbose:
              print ii," ",

          # read pixel values into a 1-D numpy array
          dataArr = numpy.array(struct.unpack_from(fmtStr, data, offset=0),
                                dtype=dtype)

          # Resize array to nx by ny pixels
          # notice order... (y,x)
          dataArr.resize((ny, nx))
          #print dataArr.shape

          # Push this image frame data onto the end of the list of images
          spedict['data'].append(dataArr)

  if verbose:
    print

  return spedict
  
def speDictToFits(spedict, outfile=None, clobber=False, verbose=False):
  """
  Given a python dictionary containing information about the SPE file
  generate a sequence of FITS files, one FITS file per image in the SPE file.

  Inputs
  spedict  --   A dictionary containing information from the SPE file
  .             as output by readSpe()
  outfile  --   root of output filename for FITS files.
  .             e.g. outfile='fitsimage'  will produce fits files
  .             named fitsimage_x000.fits
  clobber  --   Boolean.  True will overwrite any existing FITS files.
  verbose  --   Print debug messages.  

  Outputs
  fitsFileNames  --  Returns a list of the generated output filenames.

  Example usage:
  If the file called test.SPE contains 3 image frames, then
  the following code:
  >>> import piUtils
  >>> spedict = piUtils.readSpe('test.SPE')
  >>> piUtils.speDictToFits(spedict, outfile='image')
  will produce three FITS images, named:
  .   image_x001.fits
  .   image_x002.fits
  .   image_x003.fits
  """

  nframes = len(spedict['data'])
  print 'Number of frames to be saved to FITS files: ', nframes
  if nframes > 999:
    print "Warning:  speDictToFits() can only handle"
    print "up to 999 images per .spe file"
    print "Truncating the number of saved frames to 999"
    nframes = 999

  # FITS output filename
  if outfile == None:
    fitsfileroot, junk = os.path.splitext(spedict['SPEFNAME'])
  else:
    fitsfileroot = outfile
  if verbose:
    print "fitsfileroot = ", fitsfileroot

  hdrDesc = _getHeaderDescriptions()

  # Start a new HDU with image data
  fitsFilenames = []
  if verbose:
    print "writing fits image: "
  for frameNumber in range( nframes ):
  #for frameNumber in range( len(spedict['data']) ):
    frameNumberStr = "%03d" % frameNumber
    fitsFilename = fitsfileroot+"_x"+frameNumberStr+".fits"
    fitsFilenames.append(fitsFilename)
    if verbose:
      print fitsFilename,
    hdu = pyfits.PrimaryHDU(spedict['data'][frameNumber])
    for kk in hdrDesc.keys():
      hdu.header.update(kk, spedict[kk], hdrDesc[kk])
    hdu.writeto(fitsFilename, clobber=clobber)
    if verbose:
      print ""
  
  return fitsFilenames

def _getHeaderDescriptions():
  """ Defines the descriptions of the Header items that
  are populated in readSpe().  Only a few quantities (see below)
  are transferred from the SPE header into the FITS header.
  """
  hdrDesc = { 'IGAIN':"gain of intensifier, 0-255",
              'EXPOSURE':"exposure time in seconds",
              'SPEFNAME':"original SPE filename",
              'OBSDATE':"date of observation",
              'CHIPTEMP':"temperature of ICCD, in C"
              }
  
  return hdrDesc

###############################################################################
###############################################################################
####        Description of the header structure used to create piUtils      ###
###############################################################################
###############################################################################
#
#                                  WINHEAD.TXT
#
#                            $Date: 3/23/04 11:36 $
#
#                Header Structure For WinView/WinSpec (WINX) Files
#
#  The current data file used for WINX files consists of a 4100 (1004 Hex)
#  byte header followed by the data.
#
#  Beginning with Version 2.5, many more items were added to the header to
#  make it a complete as possible record of the data collection.  This includes
#  spectrograph and pulser information.  Much of these additions were accomplished
#  by recycling old information which had not been used in many versions.
#  All data files created under previous 2.x versions of WinView/WinSpec CAN
#  still be read correctly.  HOWEVER, files created under the new versions
#  (2.5 and higher) CANNOT be read by previous versions of WinView/WinSpec
#  OR by the CSMA software package.
#
#
#            ***************************************************
#
#                                    Decimal Byte
#                                       Offset
#                                    -----------
#  short   ControllerVersion              0  Hardware Version
#  short   LogicOutput                    2  Definition of Output BNC
#  WORD    AmpHiCapLowNoise               4  Amp Switching Mode
#  WORD    xDimDet                        6  Detector x dimension of chip.
#  short   mode                           8  timing mode
#  float   exp_sec                       10  alternitive exposure, in sec.
#  short   VChipXdim                     14  Virtual Chip X dim
#  short   VChipYdim                     16  Virtual Chip Y dim
#  WORD    yDimDet                       18  y dimension of CCD or detector.
#  char    date[DATEMAX]                 20  date
#  short   VirtualChipFlag               30  On/Off
#  char    Spare_1[2]                    32
#  short   noscan                        34  Old number of scans - should always be -1
#  float   DetTemperature                36  Detector Temperature Set
#  short   DetType                       40  CCD/DiodeArray type
#  WORD    xdim                          42  actual # of pixels on x axis
#  short   stdiode                       44  trigger diode
#  float   DelayTime                     46  Used with Async Mode
#  WORD    ShutterControl                50  Normal, Disabled Open, Disabled Closed
#  short   AbsorbLive                    52  On/Off
#  WORD    AbsorbMode                    54  Reference Strip or File
#  short   CanDoVirtualChipFlag          56  T/F Cont/Chip able to do Virtual Chip
#  short   ThresholdMinLive              58  On/Off
#  float   ThresholdMinVal               60  Threshold Minimum Value
#  short   ThresholdMaxLive              64  On/Off
#  float   ThresholdMaxVal               66  Threshold Maximum Value
#  short   SpecAutoSpectroMode           70  T/F Spectrograph Used
#  float   SpecCenterWlNm                72  Center Wavelength in Nm
#  short   SpecGlueFlag                  76  T/F File is Glued
#  float   SpecGlueStartWlNm             78  Starting Wavelength in Nm
#  float   SpecGlueEndWlNm               82  Starting Wavelength in Nm
#  float   SpecGlueMinOvrlpNm            86  Minimum Overlap in Nm
#  float   SpecGlueFinalResNm            90  Final Resolution in Nm
#  short   PulserType                    94  0=None, PG200=1, PTG=2, DG535=3
#  short   CustomChipFlag                96  T/F Custom Chip Used
#  short   XPrePixels                    98  Pre Pixels in X direction
#  short   XPostPixels                  100  Post Pixels in X direction
#  short   YPrePixels                   102  Pre Pixels in Y direction 
#  short   YPostPixels                  104  Post Pixels in Y direction
#  short   asynen                       106  asynchron enable flag  0 = off
#  short   datatype                     108  experiment datatype
#                                             0 =   float (4 bytes)
#                                             1 =   long (4 bytes)
#                                             2 =   short (2 bytes)
#                                             3 =   unsigned short (2 bytes)
#  short   PulserMode                   110  Repetitive/Sequential
#  WORD    PulserOnChipAccums           112  Num PTG On-Chip Accums
#  DWORD   PulserRepeatExp              114  Num Exp Repeats (Pulser SW Accum)
#  float   PulseRepWidth                118  Width Value for Repetitive pulse (usec)
#  float   PulseRepDelay                122  Width Value for Repetitive pulse (usec)
#  float   PulseSeqStartWidth           126  Start Width for Sequential pulse (usec)
#  float   PulseSeqEndWidth             130  End Width for Sequential pulse (usec)
#  float   PulseSeqStartDelay           134  Start Delay for Sequential pulse (usec)
#  float   PulseSeqEndDelay             138  End Delay for Sequential pulse (usec)
#  short   PulseSeqIncMode              142  Increments: 1=Fixed, 2=Exponential
#  short   PImaxUsed                    144  PI-Max type controller flag
#  short   PImaxMode                    146  PI-Max mode
#  short   PImaxGain                    148  PI-Max Gain
#  short   BackGrndApplied              150  1 if background subtraction done
#  short   PImax2nsBrdUsed              152  T/F PI-Max 2ns Board Used
#  WORD    minblk                       154  min. # of strips per skips
#  WORD    numminblk                    156  # of min-blocks before geo skps
#  short   SpecMirrorLocation[2]        158  Spectro Mirror Location, 0=Not Present
#  short   SpecSlitLocation[4]          162  Spectro Slit Location, 0=Not Present
#  short   CustomTimingFlag             170  T/F Custom Timing Used
#  char    ExperimentTimeLocal[TIMEMAX] 172  Experiment Local Time as hhmmss\0
#  char    ExperimentTimeUTC[TIMEMAX]   179  Experiment UTC Time as hhmmss\0
#  short   ExposUnits                   186  User Units for Exposure
#  WORD    ADCoffset                    188  ADC offset
#  WORD    ADCrate                      190  ADC rate
#  WORD    ADCtype                      192  ADC type
#  WORD    ADCresolution                194  ADC resolution
#  WORD    ADCbitAdjust                 196  ADC bit adjust
#  WORD    gain                         198  gain
#  char    Comments[5][COMMENTMAX]      200  File Comments
#  WORD    geometric                    600  geometric ops: rotate 0x01,
#                                             reverse 0x02, flip 0x04
#  char    xlabel[LABELMAX]             602  intensity display string
#  WORD    cleans                       618  cleans
#  WORD    NumSkpPerCln                 620  number of skips per clean.
#  short   SpecMirrorPos[2]             622  Spectrograph Mirror Positions
#  float   SpecSlitPos[4]               626  Spectrograph Slit Positions
#  short   AutoCleansActive             642  T/F
#  short   UseContCleansInst            644  T/F
#  short   AbsorbStripNum               646  Absorbance Strip Number
#  short   SpecSlitPosUnits             648  Spectrograph Slit Position Units
#  float   SpecGrooves                  650  Spectrograph Grating Grooves
#  short   srccmp                       654  number of source comp. diodes
#  WORD    ydim                         656  y dimension of raw data.
#  short   scramble                     658  0=scrambled,1=unscrambled
#  short   ContinuousCleansFlag         660  T/F Continuous Cleans Timing Option
#  short   ExternalTriggerFlag          662  T/F External Trigger Timing Option
#  long    lnoscan                      664  Number of scans (Early WinX)
#  long    lavgexp                      668  Number of Accumulations
#  float   ReadoutTime                  672  Experiment readout time
#  short   TriggeredModeFlag            676  T/F Triggered Timing Option
#  char    Spare_2[10]                  678  
#  char    sw_version[FILEVERMAX]       688  Version of SW creating this file
#  short   type                         704   1 = new120 (Type II)             
#                                             2 = old120 (Type I )           
#                                             3 = ST130                      
#                                             4 = ST121                      
#                                             5 = ST138                      
#                                             6 = DC131 (PentaMax)           
#                                             7 = ST133 (MicroMax/SpectroMax)
#                                             8 = ST135 (GPIB)               
#                                             9 = VICCD                      
#                                            10 = ST116 (GPIB)               
#                                            11 = OMA3 (GPIB)                
#                                            12 = OMA4                       
#  short   flatFieldApplied             706  1 if flat field was applied.
#  char    Spare_3[16]                  708  
#  short   kin_trig_mode                724  Kinetics Trigger Mode
#  char    dlabel[LABELMAX]             726  Data label.
#  char    Spare_4[436]                 742
#  char    PulseFileName[HDRNAMEMAX]   1178  Name of Pulser File with
#                                             Pulse Widths/Delays (for Z-Slice)
#  char    AbsorbFileName[HDRNAMEMAX]  1298 Name of Absorbance File (if File Mode)
#  DWORD   NumExpRepeats               1418  Number of Times experiment repeated
#  DWORD   NumExpAccums                1422  Number of Time experiment accumulated
#  short   YT_Flag                     1426  Set to 1 if this file contains YT data
#  float   clkspd_us                   1428  Vert Clock Speed in micro-sec
#  short   HWaccumFlag                 1432  set to 1 if accum done by Hardware.
#  short   StoreSync                   1434  set to 1 if store sync used
#  short   BlemishApplied              1436  set to 1 if blemish removal applied
#  short   CosmicApplied               1438  set to 1 if cosmic ray removal applied
#  short   CosmicType                  1440  if cosmic ray applied, this is type
#  float   CosmicThreshold             1442  Threshold of cosmic ray removal.  
#  long    NumFrames                   1446  number of frames in file.         
#  float   MaxIntensity                1450  max intensity of data (future)    
#  float   MinIntensity                1454  min intensity of data (future)    
#  char    ylabel[LABELMAX]            1458  y axis label.                     
#  WORD    ShutterType                 1474  shutter type.                     
#  float   shutterComp                 1476  shutter compensation time.        
#  WORD    readoutMode                 1480  readout mode, full,kinetics, etc  
#  WORD    WindowSize                  1482  window size for kinetics only.    
#  WORD    clkspd                      1484  clock speed for kinetics & frame transfer
#  WORD    interface_type              1486  computer interface                
#                                             (isa-taxi, pci, eisa, etc.)             
#  short   NumROIsInExperiment         1488  May be more than the 10 allowed in
#                                             this header (if 0, assume 1)            
#  char    Spare_5[16]                 1490                                    
#  WORD    controllerNum               1506  if multiple controller system will
#                                             have controller number data came from.  
#                                             this is a future item.                  
#  WORD    SWmade                      1508  Which software package created this file 
#  short   NumROI                      1510  number of ROIs used. if 0 assume 1.      
#
#                                      
#-------------------------------------------------------------------------------
#
#                        ROI entries   (1512 - 1631)
#
#  struct ROIinfo                           
#  {                                                   
#    WORD  startx                            left x start value.               
#    WORD  endx                              right x value.                    
#    WORD  groupx                            amount x is binned/grouped in hw. 
#    WORD  starty                            top y start value.                
#    WORD  endy                              bottom y value.                   
#    WORD  groupy                            amount y is binned/grouped in hw. 
#  } ROIinfoblk[ROIMAX]                   
#                                            ROI Starting Offsets:     
#                                                 
#                                              ROI  1  =  1512     
#                                              ROI  2  =  1524     
#                                              ROI  3  =  1536     
#                                              ROI  4  =  1548     
#                                              ROI  5  =  1560     
#                                              ROI  6  =  1572     
#                                              ROI  7  =  1584     
#                                              ROI  8  =  1596     
#                                              ROI  9  =  1608     
#                                              ROI 10  =  1620
#
#-------------------------------------------------------------------------------
#
#  char    FlatField[HDRNAMEMAX]       1632  Flat field file name.       
#  char    background[HDRNAMEMAX]      1752  background sub. file name.  
#  char    blemish[HDRNAMEMAX]         1872  blemish file name.          
#  float   file_header_ver             1992  version of this file header 
#  char    YT_Info[1000]               1996-2995  Reserved for YT information
#  long    WinView_id                  2996  == 0x01234567L if file created by WinX
#
#-------------------------------------------------------------------------------
#
#                        START OF X CALIBRATION STRUCTURE (3000 - 3488)
#
#  double  offset                      3000  offset for absolute data scaling
#  double  factor                      3008  factor for absolute data scaling
#  char    current_unit                3016  selected scaling unit           
#  char    reserved1                   3017  reserved                        
#  char    string[40]                  3018  special string for scaling      
#  char    reserved2[40]               3058  reserved                        
#  char    calib_valid                 3098  flag if calibration is valid    
#  char    input_unit                  3099  current input units for         
#                                            "calib_value"                  
#  char    polynom_unit                3100  linear UNIT and used            
#                                            in the "polynom_coeff"         
#  char    polynom_order               3101  ORDER of calibration POLYNOM    
#  char    calib_count                 3102  valid calibration data pairs    
#  double  pixel_position[10]          3103  pixel pos. of calibration data  
#  double  calib_value[10]             3183  calibration VALUE at above pos  
#  double  polynom_coeff[6]            3263  polynom COEFFICIENTS            
#  double  laser_position              3311  laser wavenumber for relativ WN 
#  char    reserved3                   3319  reserved                        
#  BYTE    new_calib_flag              3320  If set to 200, valid label below
#  char    calib_label[81]             3321  Calibration label (NULL term'd) 
#  char    expansion[87]               3402  Calibration Expansion area      
#
#-------------------------------------------------------------------------------
#
#                        START OF Y CALIBRATION STRUCTURE (3489 - 3977)
#
#  double  offset                      3489  offset for absolute data scaling
#  double  factor                      3497  factor for absolute data scaling
#  char    current_unit                3505  selected scaling unit           
#  char    reserved1                   3506  reserved                        
#  char    string[40]                  3507  special string for scaling      
#  char    reserved2[40]               3547  reserved                        
#  char    calib_valid                 3587  flag if calibration is valid    
#  char    input_unit                  3588  current input units for         
#                                            "calib_value"                   
#  char    polynom_unit                3589  linear UNIT and used            
#                                            in the "polynom_coeff"          
#  char    polynom_order               3590  ORDER of calibration POLYNOM    
#  char    calib_count                 3591  valid calibration data pairs    
#  double  pixel_position[10]          3592  pixel pos. of calibration data  
#  double  calib_value[10]             3672  calibration VALUE at above pos  
#  double  polynom_coeff[6]            3752  polynom COEFFICIENTS            
#  double  laser_position              3800  laser wavenumber for relativ WN 
#  char    reserved3                   3808  reserved                        
#  BYTE    new_calib_flag              3809  If set to 200, valid label below
#  char    calib_label[81]             3810  Calibration label (NULL term'd) 
#  char    expansion[87]               3891  Calibration Expansion area      
#
#                         END OF CALIBRATION STRUCTURES
#
#-------------------------------------------------------------------------------
#
#  char    Istring[40]                 3978  special intensity scaling string
#  char    Spare_6[25]                 4018  
#  BYTE    SpecType                    4043  spectrometer type (acton, spex, etc.)
#  BYTE    SpecModel                   4044  spectrometer model (type dependent)
#  BYTE    PulseBurstUsed              4045  pulser burst mode on/off
#  DWORD   PulseBurstCount             4046  pulser triggers per burst
#  double  ulseBurstPeriod             4050  pulser burst period (in usec)
#  BYTE    PulseBracketUsed            4058  pulser bracket pulsing on/off
#  BYTE    PulseBracketType            4059  pulser bracket pulsing type
#  double  PulseTimeConstFast          4060  pulser slow exponential time constant (in usec)
#  double  PulseAmplitudeFast          4068  pulser fast exponential amplitude constant
#  double  PulseTimeConstSlow          4076  pulser slow exponential time constant (in usec)
#  double  PulseAmplitudeSlow          4084  pulser slow exponential amplitude constant
#  short   AnalogGain;                 4092  analog gain
#  short   AvGainUsed                  4094  avalanche gain was used
#  short   AvGain                      4096  avalanche gain value
#  short   lastvalue                   4098  Always the LAST value in the header
#
#                         END OF HEADER
#
#-------------------------------------------------------------------------------
#
#                                      4100  Start of Data
#
#
#
#        ***************************** E.O.F.*****************************
#
#
#
#
#  Definitions of array sizes:
#  ---------------------------
#
#    HDRNAMEMAX  = 120     Max char str length for file name
#    USERINFOMAX = 1000    User information space
#    COMMENTMAX  = 80      User comment string max length (5 comments)
#    LABELMAX    = 16      Label string max length
#    FILEVERMAX  = 16      File version string max length
#    DATEMAX     = 10      String length of file creation date string as ddmmmyyyy\0
#    ROIMAX      = 10      Max size of roi array of structures
#    TIMEMAX     = 7       Max time store as hhmmss\0
#
#
#
#  Custom Data Types used in the structure:
#  ----------------------------------------
#
#    BYTE = unsigned char
#    WORD = unsigned short
#    DWORD = unsigned long
#
#
#
#
#  READING DATA:
#  -------------
#
#    The data follows the header beginning at offset 4100.
#
#    Data is stored as sequential points.
#    
#    The X, Y and Frame dimensions are determined by the header.
#
#      The X dimension of the stored data is in "xdim" ( Offset 42  ).
#      The Y dimension of the stored data is in "ydim" ( Offset 656 ).
#      The number of frames of data stored is in "NumFrames" ( Offset 1446 ).
#
#    The size of a frame (in bytes) is:
#
#      One frame size = xdim x ydim x (datatype Offset 108)
#
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#######################################
#### Lightly tested routines below ####
#######################################
###############################################################################
###############################################################################


#def speDictToFitsMultiExt(spedict, outfile=None, clobber=False, verbose=False):
#  """ Given an SPE file containing multiple exposures, create
#      a single, multi-extension FITS file with one image per HDU """
#    
#  # FITS output filename
#  if outfile == None:
#    fitsfileroot, junk = os.path.splitext(spedict['SPEFNAME'])
#    fitsFilename = fitsfileroot+".fits"
#  if verbose:
#    print "fitsFilename = ", fitsFilename
#    
#  hdrDesc = getHeaderDescriptions()
#
#  # Start a new HDU with image data
#  imglist = []
#  if verbose:
#    print "writing frame: ",
#  for frameNumber in range( len(spedict['data']) ):
#    if verbose:
#      print frameNumber, " ", 
#    if frameNumber == 0:
#      # Set up the primary image (the first frame)
#      # and header fields
#      hdu = pyfits.PrimaryHDU(spedict['data'][frameNumber])
#      for kk in hdrDesc.keys():
#        hdu.header.update(kk, spedict[kk], hdrDesc[kk])
#      imglist.append( hdu )
#    else:
#      imglist.append( pyfits.ImageHDU(spedict['data'][frameNumber]) )
#  if verbose:
#    print ""
#  
#  hdulist = pyfits.HDUList(imglist)
#      
#  hdulist.writeto(fitsFilename, clobber=clobber)
#
#  return fitsFilename

#def readHeaderCfg(cfgfile, verbose=False, comment='#'):
#  """ sample cfgfile is
#
#  # FITSHDRName  SPEHdrFmt offset %% description
#  IGAIN h 148 %% intensifier gain, 0-255
#
#  """
#  
#  descSep = '%%'
#  hdrCfg = {}
#  cfile = open(cfgfile, 'r')
#  for line in cfile:
#    line = line.strip()
#    if line.startswith(comment):
#      continue
#
#    hdr, description = line.split(descSep)
#    description = description.strip()
#
#    hdrName, hdrFmt, offset = hdr.split()
#    offset = int(offset)
#
#    hdrCfg[hdrName] = {'offset':offset, 'fmt':hdrFmt, 'description':description}
#
#  return hdrCfg

#def makeHeaderDict(spefilename, configfile, verbose=False):
#  """ external config file dictates what SPE header info
#  makes it into FITS file
#  """
#
#  cfgDict = readHeaderCfg(configfile)
#  print cfgDict
#
#  # open SPE file as binary input
#  spe = open(spefilename, "rb")
#  
#  # Header length is a fixed number
#  nBytesInHeader = 4100
#  # Read the entire header
#  header = spe.read(nBytesInHeader)
#
#  hdrDict = {}
#
#  for key in cfgDict.keys():
#    val = struct.unpack_from(cfgDict[key]['fmt'], header,
#                             offset=cfgDict[key]['offset'])[0]
#    hdrDict[key] = {'val':val, 'comment':cfgDict[key]['description']}
#
#  # then do the mandatory ones
#  mandatoryDict = {}
#  mandatoryDict['SPEDTYPE'] = {'offset':108, 'fmt':'h', 'description':'SPE file image data type'}
#  mandatoryDict['NX'] = {'offset':42, 'fmt':'H', 'description':'Number of x pixels'}
#  mandatoryDict['NY'] = {'offset':656, 'fmt':'H', 'description':'Number of y pixels'}
#  mandatoryDict['NFRAMES'] = {'offset':1446, 'fmt':'l', 'description':'Number of frames in original SPE file'}
#  
#  if verbose:
#    print hdrDict
#  spe.close()

#def getDataType(header, offset=108):
#
#  data_type = struct.unpack_from("h", header, offset=offset)[0]
#
#  # Determine the data type format string for
#  # upcoming struct.unpack_from() calls
#  if data_type == 0:
#    # float (4 bytes)
#    dataTypeStr = "f"  #untested
#    bytesPerPixel = 4
#  elif data_type == 1:
#    # long (4 bytes)
#    dataTypeStr = "l"  #untested
#    bytesPerPixel = 4
#  elif data_type == 2:
#    # short (2 bytes)
#    dataTypeStr = "h"  #untested
#    bytesPerPixel = 2
#  elif data_type == 3:  
#    # unsigned short (2 bytes)
#    dataTypeStr = "H"  # 16 bits in python on intel mac
#    bytesPerPixel = 2
#    dtype = "int32"  # for numpy.array().
#    # other options include:
#    # IntN, UintN, where N = 8,16,32 or 64
#    # and Float32, Float64, Complex64, Complex128
#    # but need to verify that pyfits._ImageBaseHDU.ImgCode cna handle it
#    # right now, ImgCode must be float32, float64, int16, int32, int64 or uint8
#  else:
#    print "unknown data type"
#    print "returning..."
#    return
#
#  return (dataTypeStr, bytesPerPixel, dtype)
