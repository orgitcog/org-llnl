#!/usr/bin/perl
package SL8500toACSLS;

# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Geoff Cleary <gcleary@llnl.gov>.
# LLNL-CODE-734258
#
# All rights reserved.
# This file is part of STK Address Converter. For details, see
# https://github.com/LLNL/STKAddressConverter. Licensed under the
# Apache License, Version 2.0 (the “Licensee”); you may not use
# this file except in compliance with the License. You may
# obtain a copy of the License at:
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the license.


use strict;
use English;



################################################################################
#
#  sub ConvertSL8500DriveAddrToACSLS (
#      $Library,                  <IN>
#      $Rail,                     <IN>
#      $Column,                   <IN>
#      $Side,                     <IN>
#      $Row )                     <IN>
#
#   $Library - the 1st digit in an SL8500 drive address
#   $Rail    - the 2nd digit in an SL8500 drive address
#   $Column  - the 3rd digit in an SL8500 drive address
#   $Side    - the 4th digit in an SL8500 drive address
#   $Row     - the 5th digit in an SL8500 drive address
#
#  Outputs
#      Success - a string of this form
#                "LSM,1,DriveID"
#      Failure - undef
#
#  Limitations
#      The ACS ID is not encoded in the SL8500 drive address and so this
#      function does not supply the ACS ID as part of its output.
#
#  Notes
#
################################################################################

sub ConvertSL8500DriveAddrToACSLS ( $$$$$ )
{
	#
	##  Grab the input arguments;
	my $Library = shift @_;
	my $Rail    = shift @_;
	my $Column  = shift @_;
	my $Side    = shift @_;
	my $Row     = shift @_;


	#
	##  Local variables.
	my $ArrayRow;
	my $ArrayColumn;
	my $DriveID;
	my $LSM;
	my @ACSLSDriveIDs = ( [12, 8, 4, 0],
	                      [13, 9, 5, 1],
						  [14,10, 6, 2],
						  [15,11, 7, 3] );



	#
	## Validate the inputs.
	if ( $Library < 1 )
	{
		return undef;
	}

	if ( ($Rail < 1) || ($Rail > 4) )
	{
		return undef;
	}

	if ( ($Column < -2) || ($Column > 2) || ($Column == 0) )
	{
		return undef;
	}

	if ( $Side != 1 )
	{
		return undef;
	}

	if ( ($Row < 1) || ($Row > 4) )
	{
 		return undef;
	}


	#
	##  Calculate the ACSLS LSM ID.
	$LSM = ( (4 * ($Library - 1)) + $Rail) - 1;
	

	#
	##  Convert SL8500 row to Perl array row
	$ArrayRow = $Row - 1;


	#
	##  Convert the SL8500 rolumn to Perl array rolumn
	##  SL8500 rolumns can be one of four values:
    ##    2  1 -1 -2
	##  and we need to translate them into a valid Perl array rolumn to index
	##  into the array defined at the top of the function.  Thus, multiplying by
	##  -1 and adding either 2 or 1 if negative or positive (respectively), we
	##  translate the values into
	##    0  1  2  3
	$Column *= -1;
	if ( $Column < 0 )
	{
		$ArrayColumn = $Column + 2;
	}
	elsif ( $Column > 0 )
	{
		$ArrayColumn = $Column + 1;
	}
	else
	{
		#
		## $Column is equal to zero.  This cannot happen.  Abort.
		print STDERR ("Internal error: internal column representation is " .
		              "incorrect\n.");
		exit ( 1 );
	}


	#
	##  Now that we have a row and column for the drive ID array, retrieve the ID.
	$DriveID = $ACSLSDriveIDs[$ArrayRow][$ArrayColumn];


	return ( "$LSM,1,$DriveID" );

} # ConvertSL8500DriveAddrToACSLS

return ( 1 );
