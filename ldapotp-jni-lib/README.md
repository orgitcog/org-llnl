LdapOtpNativeInterface
=========================
by Tim Meier, [meier3@llnl.gov](mailto:meier3@llnl.gov)

**libLdapOtpJni** is a Java Native Interface for `libldapotp` and `libotp_auth_client`

Released under the GNU LGPL, `LLNL-CODE-673346`.  See the `LICENSE`
file for details.

Overview
-------------------------

This native C library is an ldapotp native wrapper package.  It is designed to be dynamically
loaded by a client Java application that needs to authenticate using LDAP OTP.  Typically
this is done with the **JLdapOtpInterface.jar** library.

Installation Location
-------------------------
The standard installation is located in `/usr/lib64/ldapotp` or simply `/lib64/ldaptop`.

JLdapOtpInterface
-------------------------
A java application that needs to authenticate a user via OTP, would use the classes and methods
within the **JLdapOtpInterface** package.  This package, in turn, loads the **LdapOtpNativeInterace**
library, which links to the standard LLNL OTP authentication libraries (`libldaptop` & `libotp_auth_client`).


