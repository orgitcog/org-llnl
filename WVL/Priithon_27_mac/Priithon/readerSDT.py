from Priithon.all import N, F, U

# reading Becker & Hickl SDT files
# from time resolved fluorescence spectroscopy
#      and FLIM experiments
# ref SPC_data_file_structure.h

dtype_bhfile_header = N.dtype([
        ('revision', N.int16),   #; // software revision number  (lower 4 bits >= 10(decimal))
        ('info_offs', N.int32),  #; // offset of the info part which contains general text 
                                 #  //   information (Title, date, time, contents etc.)
        ('info_length', N.int16),#; // length of the info part
        ('setup_offs', N.int32), #; // offset of the setup text data 
                                 #  //   (system parameters, display parameters, trace parameters etc.)
        ('setup_length', N.int16),#;// length of the setup data
        ('data_block_offs', N.int32),#; offset of the first data block 
        ('no_of_data_blocks', N.int16),#;no_of_data_blocks valid only when in 0 .. 0x7ffe range,
                                       #// if equal to 0x7fff  the  field 'reserved1' contains 
                                       #//     valid no_of_data_blocks
        ('data_block_length', N.int32),#// length of the longest block in the file
        ('meas_desc_block_offs', N.int32),#;  // offset to 1st. measurement description block 
                                             #//   (system parameters connected to data blocks)
        ('no_of_meas_desc_blocks', N.int16),#;  // number of measurement description blocks
        ('meas_desc_block_length', N.int16),#;  // length of the measurement description blocks
        ('header_valid', N.uint16),         #;   // valid: 0x5555, not valid: 0x1111
        ('reserved1', N.uint32),            #;      // reserved1 now contains no_of_data_blocks
        ('reserved2', N.uint16),     
        ('chksum', N.uint16),               #;      // checksum of file header
])

BH_HEADER_CHKSUM     = 0x55aa
BH_HEADER_NOT_VALID  = 0x1111
BH_HEADER_VALID      = 0x5555



dtype_BHFileBlockHeader = N.dtype([
   ('block_no', N.int16),#;   // number of the block in the file
                         #// valid only  when in 0 .. 0x7ffe range, otherwise use lblock_no field
                         #// obsolete now, lblock_no contains full block no information
   ('data_offs', N.int32),#;       // offset of the data block from the beginning of the file
   ('next_block_offs', N.int32),#; // offset to the data block header of the next data block
   ('block_type', N.uint16),#;      // see block_type defines below
   ('meas_desc_block_no', N.int16),#; // Number of the measurement description block 
                                      #//    corresponding to this data block
   ('lblock_no', N.uint32),#;       // long block_no - see remarks below 
   ('block_length', N.uint32),#;    // reserved2 now contains block( set ) length
])


dtype_MeasStopInfo = N.dtype([ #           // information collected 
  ('status', N.uint16),  #// last SPC_test_state return value ( status )
  ('flags',  N.uint16),   #// scan clocks bits 2-0( frame, line, pixel), 
                         #  //  rates_read - bit 15
  ('stop_time', N.float32),#   time from start to  - disarm ( simple measurement )
                           #// or to the end of the cycle (for complex measurement )
  ('cur_step', N.int32),  #     // current step  ( if multi-step measurement )
  ('cur_cycle', N.int32),#     // current cycle (accumulation cycle in FLOW mode ) -
                          # //  ( if multi-cycle measurement ) 
  ('cur_page', N.int32),#;         // current measured page
  ('min_sync_rate', N.float32),#;    // minimum rates during the measurement
  ('min_cfd_rate', N.float32),#;     //   ( -1.0 - not set )
  ('min_tac_rate', N.float32),#;
  ('min_adc_rate', N.float32),#;
  ('max_sync_rate', N.float32),#;    // maximum rates during the measurement
  ('max_cfd_rate', N.float32),#;     //   ( -1.0 - not set )
  ('max_tac_rate', N.float32),#;
  ('max_adc_rate', N.float32),#;
  ('reserved1', N.int32),#;
  ('reserved2', N.float32),#;
  ])

dtype_MeasFCSInfo = N.dtype([ #   // information collected when FIFO measurement is finished
  ('chan', N.uint16),#;               // routing channel number
  ('fcs_decay_calc', N.uint16),#;     // defines which histograms were calculated
                      #// bit 0 = 1 - decay curve calculated
                      #// bit 1 = 1 - fcs   curve calculated
                      #// bit 2 = 1 - FIDA  curve calculated
                      #// bit 3 = 1 - FILDA curve calculated
                      #// bit 4 = 1 - MCS curve calculated
                      #// bit 5 = 1 - 3D Image calculated
                      #// bit 6 = 1 - MCSTA curve calculated
                      #// bit 7 = 1 - 3D MCS Image calculated
  ('mt_resol', N.uint32),#;           // macro time clock in 0.1 ns units
  ('cortime', N.float32),#;            // correlation time [ms] 
  ('calc_photons', N.uint32),#;       //  no of photons  
  ('fcs_points', N.int32),#;         // no of FCS values
  ('end_time', N.float32),#;           // macro time of the last photon 
  ('overruns', N.uint16),#;           // no of Fifo overruns 
              #//   when > 0  fcs curve & end_time are not valid
  ('fcs_type', N.uint16),#;   // 0 - linear FCS with log binning ( 100 bins/log )
              #// when bit 15 = 1 ( 0x8000 ) - Multi-Tau FCS 
              #//           where bits 14-0 = ktau parameter
  ('cross_chan', N.uint16),#;         // cross FCS routing channel number
  # //   when chan = cross_chan and mod == cross_mod - Auto FCS
  # //        otherwise - Cross FCS
  ('mod', N.uint16),#;                // module number
  ('cross_mod', N.uint16),#;          // cross FCS module number
  ('cross_mt_resol', N.uint32),#;     // macro time clock of cross FCS module in 0.1 ns units
  ])

dtype_MeasHISTInfo = N.dtype([
  ('fida_time', N.float32),#;          // interval time [ms] for FIDA histogram
  ('filda_time', N.float32),#;         // interval time [ms] for FILDA histogram
  ('fida_points', N.int32),#;        // no of FIDA values  
                 #   //    or current frame number ( fifo_image)
  ('filda_points', N.int32),#;       // no of FILDA values
                 #   //    or current line  number ( fifo_image)
  ('mcs_time', N.float32),#;           // interval time [ms] for MCS histogram
  ('mcs_points', N.int32),#;         // no of MCS values
                 #   //    or current pixel number ( fifo_image)
  ('cross_calc_phot', N.uint32),#;    //  no of calculated photons from cross_channel 
                 #   //    for Cross FCS histogram
  ('mcsta_points', N.uint16),#;       // no of MCS_TA values
  ('mcsta_flags', N.uint16),#;        // MCS_TA flags   bit 0 = 1 - use 'invalid' photons,
                 #   //      bit 1-2  =  marker no used as trigger
  ('mcsta_tpp', N.uint32),#;          // MCS_TA Time per point  in Macro Time units 
                 #   // time per point[s] = mcsta_tpp * mt_resol( from MeasFCSInfo)
  ('calc_markers', N.uint32),#;       // no of calculated markers for MCS_TA 
  ('reserved3', N.float64),#;
])

dtype_MeasureInfo  = N.dtype([
  ('time', N.character, 9),        # /* time of creation */
  ('date', N.character, 11),       # /* date of creation */
  ('mod_ser_no', N.character, 16), # /* serial number of the module */
  ('meas_mode', N.int16),
  ('cfd_ll', N.float32),
  ('cfd_lh', N.float32),
  ('cfd_zc', N.float32),
  ('cfd_hf', N.float32),
  ('syn_zc', N.float32),
  ('syn_fd', N.int16),
  ('syn_hf', N.float32),
  ('tac_r', N.float32),
  ('tac_g', N.int16),
  ('tac_of', N.float32),
  ('tac_ll', N.float32),
  ('tac_lh', N.float32),
  ('adc_re', N.int16),
  ('eal_de', N.int16),
  ('ncx', N.int16),
  ('ncy', N.int16),
  ('page', N.uint16),
  ('col_t', N.float32),
  ('rep_t', N.float32),
  ('stopt', N.int16),
  ('overfl', N.character, 1), ## orig 1
  ('use_motor', N.int16),
  ('steps', N.uint16),
  ('offset', N.float32),
  ('dither', N.int16),
  ('incr', N.int16),
  ('mem_bank', N.int16),
  ('mod_type', N.character, 16),   # /* module type */
  ('syn_th', N.float32),
  ('dead_time_comp', N.int16),
  ('polarity_l', N.int16),   #   2 = disabled line markers
  ('polarity_f', N.int16),
  ('polarity_p', N.int16),
  ('linediv', N.int16),      #  line predivider = 2 ** ( linediv),
  ('accumulate', N.int16),
  ('flbck_y', N.int32), 
  ('flbck_x', N.int32), 
  ('bord_u', N.int32), 
  ('bord_l', N.int32), 
  ('pix_time', N.float32),
  ('pix_clk', N.int16), 
  ('trigger', N.int16), 
  ('scan_x', N.int32), 
  ('scan_y', N.int32), 
  ('scan_rx', N.int32), 
  ('scan_ry', N.int32), 
  ('fifo_typ', N.int16),
  ('epx_div', N.int32), 
  ('mod_type_code', N.uint16), 
  ('mod_fpga_ver', N.uint16),    #  new in v.8.4
  ('overflow_corr_factor', N.float32),
  ('adc_zoom', N.int32), 
  ('cycles', N.int32),        #   cycles ( accumulation cycles in FLOW mode ) 
  ('StopInfo', dtype_MeasStopInfo), #N.character, 4*15),    # MeasStopInfo StopInfo')
  ('FCSInfo', dtype_MeasFCSInfo), #N.character, 2*19),     # MeasFCSInfoFCSInfo')              #  valid only for FIFO meas
  ('image_x', N.int32),       #  4 subsequent fields valid only for Camera mode
  ('image_y', N.int32),       #      or FIFO_IMAGE mode
  ('image_rx', N.int32), 
  ('image_ry', N.int32), 
  ('xy_gain', N.int16),       #  gain for XY ADCs ( SPC930 )
  ('dig_flags', N.int16),     #SP_MST_CLK parameter - digital flags : 
                         #  bit 0 - use or not  Master Clock (SPC140 multi-module )
                         #  bit 1 - Continuous Flow On/Off for scan modes in SPC150
  ('adc_de', N.int16),        #  ADC sample delay ( SPC-930 )
  ('det_type', N.int16),      #  detector type ( SPC-930 in camera mode )
  ('x_axis', N.int16),        #  X axis representation ( SPC-930 ) 
  ('HISTInfo', dtype_MeasHISTInfo), #N.character, 12*4),                #MeasHISTInfo  HISTInfo              #  extension of FCSInfo, valid only for FIFO meas
])

# class ndarray_inSdtFile(N.ndarray):
#     def __new__(cls, input_array, sdtInfo=None):
#         obj = N.asanyarray(input_array).view(cls)
#         obj.SDT = sdtInfo
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.SDT = getattr(obj, 'SDT', None)
#         self.__dict__.update(getattr(obj, "__dict__", {}))
class ndarray_inSdtFile(N.ndarray):
    def __new__(cls, input_array, dataHdr=None, dataMeasInfo=None, sdt=None):
        obj = N.asanyarray(input_array).view(cls)
        obj.SDT_DataHdr = dataHdr
        obj.SDT_DataMeasInfo = dataMeasInfo
        obj.SDT=sdt
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        #self.SDT = getattr(obj, 'SDT', None)
        self.SDT_DataHdr = getattr(obj, 'SDT_DataHdr', None)
        self.SDT_DataMeasInfo = getattr(obj, 'SDT_DataMeasInfo', None)
        self.SDT = getattr(obj, 'SDT', None)
        self.__dict__.update(getattr(obj, "__dict__", {}))




def open_bh_sdt(fn, mode='r'):
    """
    read Becker&Hickel SDT file into separate structures
    any error here might result from the file being corrupt
    """
    #b = file(fn, 'rb').read()
    m = N.memmap(fn, mode=mode)

    hdrEnd = dtype_bhfile_header.itemsize
    lenDBhdr = dtype_BHFileBlockHeader.itemsize

    hdr = N.rec.array(m[:hdrEnd], dtype=dtype_bhfile_header)   [0]
    #hdr = m[:hdrEnd].view(dtype_bhfile_header)

    if hdr['header_valid'] != BH_HEADER_VALID:
        print " * WARNING * SDT file file header not valid"

    infoStart = hdr['info_offs']
    infoEnd   = infoStart + hdr['info_length']
    info= m[infoStart:infoEnd]
    info= info.view(N.dtype( (N.string_, len(info)) ))     [0]

    setupStart= hdr['setup_offs']
    setupEnd  = setupStart + hdr['setup_length']
    setup = m[setupStart:setupEnd]
    setup = setup.view(N.dtype( (N.string_, len(setup)) ))    [0]
    setupTxtEnd = N.char.find(setup, '*END') + len('*END')
    setupTxt = setup[:setupTxtEnd]
    # following bytes are: '\r\n\r\n.......'
    # >>> U.hexdump(s.setup[s.setupTxtEnd:s.setupTxtEnd+21])
    # 0000  0d 0a 0d 0a 42 49 4e 5f 50 41 52 41 5f 42 45 47   ....BIN_PARA_BEG
    # 0010  49 4e 3a 00 0e                                    IN:..
    #setupBin = 

    nDBs = hdr['no_of_data_blocks']
    if nDBs == 0x7fff:
        nDBs = hdr['reserved1']

    #if nDBs != 1:
    #    print " * WARNING * SDT file has more than one data block(nDBs=%s)"%(nDBs,)

    nMDBs = hdr['no_of_meas_desc_blocks']
    mdbStart = hdr['meas_desc_block_offs']
    lenMDB = hdr['meas_desc_block_length']
    numMDB = hdr['no_of_meas_desc_blocks']
    dataMeasInfos = m[mdbStart:mdbStart+numMDB*lenMDB].view(dtype=dtype_MeasureInfo)   

    import weakref
    dbHdr  = []
    dbData = []
    db0Start = hdr['data_block_offs']
    o = db0Start
    for i in range(nDBs):
        dbh = m[o:o+lenDBhdr].view(dtype=dtype_BHFileBlockHeader)   [0]
        dbHdr.append( dbh )
        dbo = dbh['data_offs']
        dbl = dbh['block_length']
        #dbd = N.array(m[dbo:dbo+dbl], dtype=N.uint16)
        #ValueError: setting an array element with a sequence.
        dbd = m[dbo:dbo+dbl].view(dtype=N.uint16)

        #dbd.SDT_DataHdr      = dbh
        #dbd.SDT_DataMeasInfo = weakref.proxy( measInfos[ dbh['meas_desc_block_no'] ] )
        dbd = ndarray_inSdtFile(dbd, 
                                dataHdr=dbh, 
                                dataMeasInfo=dataMeasInfos[ dbh['meas_desc_block_no'] ]
                                )

        nx = dbd.SDT_DataMeasInfo['image_x']
        ny = dbd.SDT_DataMeasInfo['image_y']

        if nx>0 and ny >0:           # FLIM !!
            dbd.shape = nx,ny,-1
            dbd = dbd.T

        dbData.append(dbd)
        o = dbh['next_block_offs']
        del dbh,dbd,dbo, dbl,  nx, ny


    del o, i, weakref

    dataBlkHdrs = F.mockNDarray(*dbHdr)
    dataBlks = F.mockNDarray(*dbData)

    del dbHdr, dbData
                                          
    #return U.localsAsOneObject()
    #data = ndarray_inSdtFile(dbData[0], U.localsAsOneObject())
    #data.SDT.data = weakref.proxy( self.) # remove circular reference
    dataBlks.SDT = U.localsAsOneObject()
    import weakref
    dataBlks.SDT.dataBlks = weakref.proxy( dataBlks ) # remove circular reference

    if len(dataBlks) == 1 and dataBlks[0].ndim>1:
        # one FLIM data block
        dataBlks[0].SDT = dataBlks.SDT.dataBlks
        return dataBlks[0]
    else:
        return dataBlks



def sdt2asc(fn, ext='_pri.asc'):
    """
    load sdt file `fn`
    save asc format into `fn[:-4]+ext`
    """

    fnOut = fn[:-4]+ext

    ss = open_bh_sdt(fn)
    import string, StringIO
    ssi = StringIO.StringIO(ss.info)
    ssil = ssi.readlines()
    infoDict = dict([map(string.strip, l.split(':', 1)) for l in ssil if ':' in l])

    import os
    if os.path.exists(fnOut):
        if raw_input("overwrite file '%s' ?"%(fnOut,)) not in ('y', 'Y', 'yes', 'YES'):
            print "not converted - nothing done."
            return

    #
    #    force '\n\r' newline even on non Windows:
    #
    #     f = file(fn, "wb")
    #     # #python IRC 20090625
    #     #if you open the file in binary mode, you can do this: 
    #     old_write = f.write
    #     def write(text): 
    #         old_write(text.replace('\n', '\r\n'))
    #     f.write = write
#     class WindowsNewlines(object):
#         def __init__(self, file): 
#             self.file = file 
#         def write(self, s): 
#             #print 'write'
#             #self.__base__.write(s.replace('\n', '\r\n'))
#             self.file.write(s.replace('\n', '\r\n'))
#     f = WindowsNewlines(file(fnOut, "wb"))

# # # # #     class file_winNewline(file):
# # # # #         def write(self, s):
# # # # #             print 'hallo'
# # # # #     f = file_winNewline(fnOut, "wb")

    #
    #    force '\n\r' newline even on non Windows:
    #

    import re
    p = re.compile(' ( +)')

    f = file(fnOut, "wb")

    for k in ('Title', 'Version','Revision',
              'Date','Time','Author','Company','Contents',):
        v = infoDict[k]
        v = p.sub(' ', v) # squeeze spaces ('Version : 1  894 M' -> 'Version : 1 894 M')

        print >>f, '  %s :%s\r\n'%(k, (' '+v) if v else ''),
    print >>f, '  '+ '\r\n',
    for block in range(len(ss.dbData)):
        print >>f, "*BLOCK %d\r\n"%(block+1,),
        #print '\n'.join(map(str, ss.dbData))
        ss.dbData[block].tofile(f, "\r\n")
        print >>f, '\r\n',
        print >>f, "*END\r\n",
        
def writeAsc(fnOut, dbData, info, overwrite=False):
    """
    ask before overwrite
    
    `info` can either be a (preparsed" dict
       or a "raw" string of the info block (as it comes in the sdt file   
    """
    import os

    if not overwrite and os.path.exists(fnOut):
        if raw_input("overwrite file '%s' ?"%(fnOut,)) not in ('y', 'Y', 'yes', 'YES'):
            print "not converted - nothing done."
            return

    if isinstance(info, dict):
        infoDict = info
    else:
        import string, StringIO
        ssi = StringIO.StringIO(info)
        ssil = ssi.readlines()
        infoDict = dict([map(string.strip, l.split(':', 1)) for l in ssil if ':' in l])
        
    import re
    p = re.compile(' ( +)')

    f = file(fnOut, "wb")

    for k in ('Title', 'Version','Revision',
              'Date','Time','Author','Company','Contents',):
        v = infoDict[k]
        v = p.sub(' ', v) # squeeze spaces ('Version : 1  894 M' -> 'Version : 1 894 M')

        print >>f, '  %s :%s\r\n'%(k, (' '+v) if v else ''),
    print >>f, '  '+ '\r\n',
    for block in range(len(dbData)):
        print >>f, "*BLOCK %d\r\n"%(block+1,),
        #print '\n'.join(map(str, dbData))
        dbData[block].tofile(f, "\r\n")
        print >>f, '\r\n',
        print >>f, "*END\r\n",




def loadFLIM(fn, shape=(256,256,256)):
    ss = open_bh_sdt(fn)
    #>>> ss.dbData.shape
    #(1, 16777216)
    a = ss.dbData[0].view()
    a.shape=shape
    return a.T
