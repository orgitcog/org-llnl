"""
A simple class that reads in different solar system initial abundance files and returns various values that
the user might need, such as abundances normed to various things, ratios, delta-values, etc. Have a look at
the readme or the docstring for detailed instructions.
"""
import numpy as np
import sys, os


class IniAbu:
    """
    The class' docstring
    """
    def __init__(self, database='lodders09'):
        """
        Load this class the following way into an instant i:

        IMPORT THE CLASS:
        +++++++++++++++++

        import iniabu
        i = iniabu.IniAbu(database)

        QUERY AN ISOTOPE OR ELEMENT:
        ++++++++++++++++++++++++++++

        NOTATION:
        The notation to query an element is either a string that uses the official abbreviation (e.g., 'Fe') or an
        integer, that gives the number of protons of a given element.
        The notation to query an isotope is either a string specifying the isotope (e.g., 'Fe56') or a tuple that
        specifies (zz, aa), i.e., the number of protons and the mass number of the isotope.

        There are two querying routines, to return information on an isotope or information on an element.

        i.query_iso('Si28')

        For example returns

        [14, 'si', 28, 92.23, 922000.0]

        Here the list is the number of protons of Si, the element name, the mass number, the abundance with respect
        the other isotopes (in percent) as well as the number abundnace in the solar system, normed according to the
        database that serves as the input. Similarly, elements can be queried by typing, e.g.:

        i.query_ele('Si')

        This returns:

        [14, 'si', 28.10857, 999700.0]

        Here the entries are: Number of protons, Element name, Average mass (calculated from mass numbers and isotope
        abundances), and total sum of solar system abundance of all isotopes.

        CALCULATE BRACKET NOTATION, DELTA VALUE, RATIOS, ...:
        +++++++++++++++++++++++++++++++++++++++++++++++++++++

        Various subroutines are available to calculate delta-values, bracket notation, isotope, or element ratios.
        These function often have options, i.e., do you want the mass fraction or the number fraction (default). Please
        check out the individual doc strings of the functions. A list of the currently available functions is here:

            i.bracket_ele:      Calculates bracket notation for a given value with the solar ratio of elements
            i.bracket_iso:      Calculates bracket notation for a given value with the solar ratio of isotopes
            i.delta_ele:        Calculates delta-value for a given value with the solar element ratio
            i.delta_iso:        Calculates delta-value for a given value with the solar isotope ratio
            i.ele_ratio:        Calculates the ratio of two elements in solar composition
            i.ele_iso:          Calculates the ratio of two isotopes in solar composition

        DATABASE:
        +++++++++

        By default, the standard database is Lodders et al. (2009). However, the following other databases are possible:
        - lodders09:    Lodders et al. (2009)

        :param database:    Database to load for isotope abundances
        """
        # run script under debugging conditions
        self._debug = False

        # initialize empty arrays for input data
        self.zz = None   # list of proton numbers
        self.aa = None   # list of proton + neutron numbers
        self.ele = None   # list of element names - First letter always capitalized (check in input routine)
        self.atomp = None   # list of atom percentages, of given element, total is 100.
        self.nn = None   # list of number percentages

        # read in the correct database
        if database == 'lodders09':
            if self._debug:
                print('Reading in Lodders et al. (2009) file.')
            self._read_lodders09()

        # make sure that all the elements are in lower case only!
        for it in range(len(self.ele)):
            self.ele[it] = self.ele[it].lower()

        # sort the data after readin into a list of lists
        # [[Symbol, Z, [A1, A2, ...], [AtomP1, Atomp2, ...], [N1, N2, ...]], [...], ...]
        self.data = []
        self.elesingle = []   # elements in single vector
        self.zzsingle = []   # Z in single vector
        # initialize temporary variables
        zztmp = 0
        aatmp = []
        atomptmp = []
        nntmp = []
        for it in range(len(self.zz)):
            if self.zz[it] != zztmp:   # new entry!
                if it > 0:   # add to data, unless it's the first one
                    self.data.append([self.ele[it-1], self.zz[it-1], aatmp, atomptmp, nntmp])
                    self.elesingle.append(self.ele[it-1])
                    self.zzsingle.append(self.zz[it-1])
                # reset temporary variables
                zztmp = self.zz[it]
                aatmp = []
                atomptmp = []
                nntmp = []
            # append data
            aatmp.append(self.aa[it])
            atomptmp.append(self.atomp[it])
            nntmp.append(self.nn[it])
        # add the last data entry
        self.data.append([self.ele[len(self.ele)-1], self.zz[len(self.ele)-1], aatmp, atomptmp, nntmp])
        self.elesingle.append(self.ele[len(self.ele)-1])
        self.zzsingle.append(self.zz[len(self.ele)-1])

        # make dictionaries
        self.eledict = dict(zip(self.elesingle, range(len(self.data))))
        self.zzdict = dict(zip(self.elesingle, self.zzsingle))
        self.eledictzz = dict(zip(self.zzsingle, self.elesingle))

    # CALCULATIONS AND RETURNS
    def bracket_ele(self, value, ele1, ele2, massf=False):
        """
        Returns the bracket notation normalization for 'value', which is (N_ele1 / N_ele2)_star, normalized to the
        solar value. By default, number ratios are assuemd for the value. The definition of the bracket notation is:
            log_10(value) - log_10(ele1 / ele2)_solar
        :param value:   <float> Value that should be calculated from
        :param ele1:    <str>   Element 1, e.g., 'Si'
        :param ele2:    <str>   Element 2, e.g., 'Fe'
        :return:        <float> Bracket value for, e.g., [Si/Fe]
        """
        ratio = np.log10(value) - np.log10(self.ele_ratio(ele1, ele2, massf=massf))
        return ratio

    def bracket_iso(self, value, iso1, iso2, massf=False):
        """
        Returns the bracket notation normalization for 'value', which is (N_iso1 / N_iso2)_star, normalized to the
        solar value. By default, number ratios are assuemd for the value. The definition of the bracket notation is:
            log_10(value) - log_10(iso1 / iso2)_solar
        :param value:   <float> Value that should be calculated from
        :param iso1:    <str>   Isotope 1, e.g., 'Si30'
        :param iso2:    <str>   Isotope 2, e.g., 'Si28'
        :return:        <float> Bracket value for, e.g., [Si30/Si30]
        """
        ratio = np.log10(value) - np.log10(self.iso_ratio(iso1, iso2, massf=massf))
        return ratio

    def delta_ele(self, value, ele1, ele2, mult=1000., massf=False):
        """
        Return a delta value for the given value and the element ratios. The delta value is simply the standard
        deviation of a ratio from the solar ratio (defined via ele1 / ele2). If mult=1000. is chosen, the delta values
        are returned in per mille. However, this can be changed by changing the multiplier mult. By default, it is
        assumed that the value is given in number fraction. The definition of the
        delta value is:
            (value / (ele1 / ele2) - 1.) * mult.
        :param value:   <float> Value for which the delta value should be calculated
        :param ele1:    <str>   Element in nominator, i.e., 'Fe'
        :param ele2:    <str>   Element in denominator, i.e., 'Si'
        :param mult:    <float> What multiplier should be used? Default is 1000, i.e., return detla values in per mille
        :param massf:   <bool>  Default False. If true, number ratios are assumed
        :return:
        """
        deltaval = (value / self.ele_ratio(ele1, ele2, massf=massf) - 1.) * mult
        return deltaval

    def delta_iso(self, value, iso1, iso2, mult=1000., massf=False):
        """
        Return a delta value for the given value and the isotope ratios. The delta value is simply the standard
        deviation of a ratio from the solar ratio (defined via iso1 / iso2). If mult=1000. is chosen, the delta values
        are returned in per mille. However, this can be changed by changing the multiplier mult. By default, it is
        assumed that the value is given in number fraction. The definition of the
        delta value is:
            (value / (iso1 / iso2) - 1.) * mult.
        :param value:   <float> Value for which the delta value should be calculated
        :param iso1:    <str>   Isotope in nominator, i.e., 'Fe56'
        :param iso2:    <str>   Isotope in denominator, i.e., 'Fe56'
        :param mult:    <float> What multiplier should be used? Default is 1000, i.e., return detla values in per mille
        :param massf:   <bool>  Default False. If true, number ratios are assumed
        :return:
        """
        deltaval = (value / self.iso_ratio(iso1, iso2, massf=massf) - 1.) * mult
        return deltaval

    def ele_ratio(self, ele1, ele2, massf=False):
        """
        Returns an element ratio of ele1 / ele2. By default, a number ratio is returned. Mass ratio is returned if
        massf is True
        :param ele1:    <str>   First element, e.g., 'Fe'
        :param ele2:    <str>   Second element, e.g., 'Si'
        :param massf:   <bool>  If true, mass ratio is returned. Default is False
        :return:        <float> Element ratio
        """
        qele1 = self.query_ele(ele1)
        qele2 = self.query_ele(ele2)
        ratio = qele1[3] / qele2[3]
        if massf:
            ratio *= qele2[2] / qele1[2]
        return ratio

    def iso_ele_ratio(self, iso, ele, massf=False):
        """
        Returns an isotope to element ratio. By default, a number ratio is returned. Mass ratio is returned if massf is
        True.
        :param iso:     <str>   Isotope of interest, e.g., 'Fe56'
        :param ele:     <str>   Element to normalize to, e.g., 'Si'
        :param massf:   <bool>  Default: False  Should a mass fraction be returned?
        :return:        <float> Isotope to element ratio
        """
        qiso = self.query_iso(iso)
        qele = self.query_ele(ele)
        ratio = qiso[4] / qele[3]
        if massf:
            ratio *= qele[2] / qiso[2]
        return ratio

    def iso_ratio(self, iso1, iso2, massf=False):
        """
        Returns an isotope ratio of iso1 / iso2. By default, a number ratio is returned. Mass ratio is returned if
        massf is True
        :param iso1:    <str> Isotope nominator, e.g., 'Fe56'
        :param iso2:    <str> Isotope denominator, e.g., 'Fe56'
        :param massf:   <bool> Return mass ratio? The default is False (i.e., return number ratio)
        :return:        <float> Ratio of istotope abundances
        """
        quiso1 = self.query_iso(iso1)
        quiso2 = self.query_iso(iso2)
        ratio = quiso1[4] / quiso2[4]
        # mass fraction if wanted
        if massf:
            ratio *= quiso2[2] / quiso1[2]
        return ratio

    # QUERIES OF DATABASES
    def query_ele(self, ele):
        """
        Queries an element and returns its properties
        :param ele:     <str>   Element name, e.g., 'Fe'
        :return:        <list>  [zz, ele, aa_avg, total_nn_of_all_isotopes]
                                Z of element, the element name, average mass, total abundance of element
        """
        # get element index and check if it is available
        try:
            eleindex = self.eledict[ele.lower()]
        except KeyError:
            _err_notfound(ele)
        # sanity check if debug is turned on
        if self._debug:
            if np.abs(np.sum(self.data[eleindex][3]) - 100.) > 0.000001:
                print('Element ' + ele + ' does not sum to 1 but to ' + str(np.sum(self.data[eleindex][3])) + \
                      '! Check the database.')

        # calculate the average mass
        avg_mass = np.sum(np.array(self.data[eleindex][2]) * np.array(self.data[eleindex][3])) / 100.

        # now return the result
        return [self.zzsingle[eleindex], ele, avg_mass, np.sum(self.data[eleindex][4])]

    def query_iso(self, iso):
        """
        Queries an isotope and returns its properties
        :param iso:     <str, tuple>   Isotope as string, e.g., 'Fe56'
                                       Isotope can also be given as a tuple, e.g., (zz, aa)
        :return:        <list>  [zz, ele, aa, atomp, nn]
        """
        # parse input if given as string
        if type(iso) == str:
            # remove spaces, dashes, and underlines - to deal with various user input
            iso = str(iso.replace(' ', ''))
            iso = str(iso.replace('-', ''))
            iso = str(iso.replace('_', ''))
            # now parse the input
            for it in range(len(iso)):
                try:
                    aa = int(iso[it:len(iso)])
                    ele = iso[0:it].lower()
                    break
                except ValueError:
                    pass
        # parse input if given as tuple
        elif type(iso) == tuple:
            aa = iso[1]
            try:
                ele = self.eledictzz[iso[0]]
            except KeyError:
                _err_notfound(iso)
        else:
            sys.exit('You must specify the element as a string (e.g., \'Fe60\' or as a tuple of number of protons (zz)'
                     ' and total mass number (aa), such that the tuple looks like (zz, aa). Your input was invalid.')

        # check if element and isotope exists
        try:
            eleindex = self.eledict[ele]
        except KeyError:
            _err_notfound(iso)
        isoindex = -1
        for it in range(len(self.data[eleindex][2])):
            if aa == self.data[eleindex][2][it]:
                isoindex = it
                break
        if isoindex == -1:
            _err_notfound(iso)

        # now grab the stuff we want to return and return it
        return [self.zzdict[ele], ele, aa, self.data[eleindex][3][isoindex], self.data[eleindex][4][isoindex]]

    # READERS OF DATABASES
    def _read_lodders09(self):
        """
        Reads in the lodders09.dat file in the data folder and puts it in the right format
        :return:
        """
        # open file and read in
        f = self._open_file('lodders09.dat')
        # check if none
        if self._debug:
            if f is None:
                print('Error: Could not read in lodders09.dat file')

        # make data from file
        datall = f.read().split('\n')
        f.close()

        # header
        # header = datall[0].split()

        # data
        data = []
        for it in range(1, len(datall)):
            data.append(datall[it].split())

        # now put all into lists and then make numpy arrays
        zz = []   # Z of element
        ele = []   # element
        aa = []   # a of element
        atomp = []   # atom percentage
        nn = []   # number abundance

        for it in range(len(data)):
            if len(data[it]) == 5:   # check that there are 5 entries exactly
                zz.append(int(data[it][0]))
                ele.append(data[it][1])
                aa.append(int(data[it][2]))
                atomp.append(float(data[it][3]))
                nn.append(float(data[it][4]))

        # make np arrays and write to class
        self.zz = np.array(zz)
        self.aa = np.array(aa)
        self.ele = ele
        self.atomp = atomp
        self.nn = nn

    def _open_file(self, fname):
        """
        Opens a file and returns its content. Checks if it is in the pip installation folder or local in a subfolder
        :param fname:   <str> Filename of file to open
        :return:        File
        """
        # error toggles
        globalerr = False
        localerr = False
        # find the global file from PyPi package
        try:
            install_path = os.path.dirname(os.path.realpath(__file__))
            full_fname = os.path.join(install_path, 'iniabu_data', fname)
            f = open(full_fname, 'r')
            return f
        except FileNotFoundError:
            # all failed, give back None
            globalerr = True
        # if user is local
        try:
            f = open('iniabu_data/' + fname, 'r')
            return f
        except FileNotFoundError:
            localerr = True

        # raiser errors
        if globalerr:
            print('Global file ' + fname + ' not found. Try reinstalling the PyPi package')
        if localerr:
            print('Local file ' + fname + ' not found. Make sure you are running local or that the global package is '
                                          'properly installed.')
        if globalerr or localerr:
            sys.exit('Quitting since read in file could not be found')


# ### RAISE ERROR MESSAGES AND EXIT
# ERRORS
def _err_notfound(errname):
    """
    Displays an error message that the string 'ele' was not found
    :param errname:     <str> Name of what element was not found
    :return:            None, just raise an error
    """
    sys.exit('ERROR: Could not find ' + errname)


# Call if main
if __name__ == "__main__":
    i = IniAbu()
