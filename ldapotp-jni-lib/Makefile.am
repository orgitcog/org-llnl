JAVAHOME=/usr/lib/jvm/java

INCLUDES = -I$(includedir)/ldapotp -I$(JAVAHOME)/include -I$(JAVAHOME)/include/linux $(GLIB_CFLAGS)

lib_LTLIBRARIES = src/libLdapOtpJni.la

DBGFLAGS = -g3

src_libLdapOtpJni_la_CFLAGS = -O0 -Wall -fmessage-length=0 $(DBGFLAGS)

if HAVE_LD_VERSION_SCRIPT
    libLdapOtpJni_version_script = -Wl,--version-script=$(srcdir)/libLdapOtpJni.map
else
    libLdapOtpJni_version_script =
endif

src_libLdapOtpJni_la_SOURCES = src/jni_LdapOtpNativeInterface.c src/jni_NativeUtils.c src/jni_PeerClass.c
src_libLdapOtpJni_la_LDFLAGS =  -shared -fPIC -version-info $(libLdapOtpJni_api_version) -export-dynamic $(libLdapOtpJni_version_script)
src_libLdapOtpJni_la_LIBADD = -L$(libdir) -L/usr/lib64/ldapotp -lotp_auth_client $(GLIB_LIBS)
src_libLdapOtpJni_la_DEPENDENCIES = $(srcdir)/libLdapOtpJni.map

sbin_SCRIPTS =

man_MANS = man/LdapOtpJni.8


# headers are distributed as part of the include dir
EXTRA_DIST = $(srcdir)/llnl-ldapotp-clt-jni-auth.spec.in \
				 $(srcdir)/libLdapOtpJni.map \
				 $(srcdir)/src/*.h \
				 $(srcdir)/config \
				 $(srcdir)/ChangeLog \
				 $(srcdir)/INSTALL \
				 $(srcdir)/COPYING \
				 $(srcdir)/LICENSE \
				 $(srcdir)/README.md \
			    $(srcdir)/META \
			    $(srcdir)/DISCLAIMER \
				 $(man_MANS)

dist-hook: llnl-ldapotp-clt-jni-auth.spec
	cp llnl-ldapotp-clt-jni-auth.spec $(distdir)


