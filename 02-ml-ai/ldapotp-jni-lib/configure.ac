dnl Process this file with autoconf to produce a configure script.

AC_INIT(llnl-ldapotp-clt-jni-auth-libs, 1.0.0, meier3@llnl.gov)
AC_META
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_SRCDIR([META])
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE($PROJECT, $VERSION)

dnl Checks for programs
AC_PROG_CC
AC_PROG_GCC_TRADITIONAL
AC_PROG_LIBTOOL

dnl set up a #define for the code to find the config file.
CONF_TMP1="`eval echo ${sysconfdir}`"
CONF_TMP2="`echo $CONF_TMP1 | sed 's/^NONE/$ac_default_prefix/'`"
AC_DEFINE_UNQUOTED(CONF_FILE_PREFIX, ["$CONF_TMP2"], [Config Directory])

dnl LDAPTOTP headers are in $includedir/ldapotp
INC_TMP1="`eval echo ${includedir}/ldapotp`"
INC_TMP2="`echo $INC_TMP1 | sed 's/^NONE/$ac_default_prefix/'`"
CFLAGS="$CFLAGS -I`eval echo $INC_TMP2`"
CPPFLAGS="$CPPFLAGS -I`eval echo $INC_TMP2`"

dnl Checks for libraries
AC_CHECK_LIB(pthread, pthread_mutex_init, [],
	AC_MSG_ERROR([pthread_mutex_init() not found.  llnl-ldapotp-jni-lib requires libpthread.]))

dnl the library version
libLdapOtpJni_api_version=1:1:0
AC_SUBST(libLdapOtpJni_api_version)

dnl We use --version-script with ld if possible
AC_CACHE_CHECK(whether ld accepts --version-script, ac_cv_version_script,
    if test -n "`$LD --help < /dev/null 2>/dev/null | grep version-script`"; then
        ac_cv_version_script=yes
    else
        ac_cv_version_script=no
    fi)

AM_CONDITIONAL(HAVE_LD_VERSION_SCRIPT, test "$ac_cv_version_script" = "yes")

# we have to revive the env CFLAGS as some how they are being overwritten...
# see http://sources.redhat.com/automake/automake.html#Flag-Variables-Ordering
# for why they should NEVER be modified by the configure to allow for user
# overrides.
CFLAGS=$ac_env_CFLAGS_value

dnl Make appropriate substitution for sysconfdir for use in scripts
dnl Must expand nested unquoting
CONFFILEPREFIX_TMP1="`eval echo ${sysconfdir}`"
CONFFILEPREFIX_TMP2="`echo $CONFFILEPREFIX_TMP1 | sed 's/^NONE/$ac_default_prefix/'`"
CONFFILEPREFIX="`eval echo $CONFFILEPREFIX_TMP2`"
AC_SUBST(CONFFILEPREFIX)

AC_CONFIG_FILES([ldap_otp_jni_version.h \
		Makefile \
		llnl-ldapotp-clt-jni-auth.spec \
		])
AC_OUTPUT
