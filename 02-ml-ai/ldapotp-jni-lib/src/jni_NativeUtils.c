/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Timothy Meier, meier3@llnl.gov, All rights reserved.
 * LLNL-CODE-673346
 *
 * This file is part of the OpenSM Monitoring Service (OMS) package.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * OUR NOTICE AND TERMS AND CONDITIONS OF THE GNU GENERAL PUBLIC LICENSE
 *
 * Our Preamble Notice
 *
 * A. This notice is required to be provided under our contract with the U.S.
 * Department of Energy (DOE). This work was produced at the Lawrence Livermore
 * National Laboratory under Contract No.  DE-AC52-07NA27344 with the DOE.
 *
 * B. Neither the United States Government nor Lawrence Livermore National
 * Security, LLC nor any of their employees, makes any warranty, express or
 * implied, or assumes any liability or responsibility for the accuracy,
 * completeness, or usefulness of any information, apparatus, product, or
 * process disclosed, or represents that its use would not infringe privately-
 * owned rights.
 *
 * C. Also, reference herein to any specific commercial products, process, or
 * services by trade name, trademark, manufacturer or otherwise does not
 * necessarily constitute or imply its endorsement, recommendation, or favoring
 * by the United States Government or Lawrence Livermore National Security,
 * LLC. The views and opinions of authors expressed herein do not necessarily
 * state or reflect those of the United States Government or Lawrence Livermore
 * National Security, LLC, and shall not be used for advertising or product
 * endorsement purposes.
 *
 * jni_NativeUtils.c
 *
 *  Created on: Jun 2, 2011
 *      Author: meier3
 */

#include <jni.h>
#include <ldapotp/ldapotp-client.h>
#include <ldapotp/otp_auth_client.h>
#include <stdio.h>
#include <unistd.h>

#include "ldap_otp_jni_version.h"

#include "jni_PeerClass.h"
#include "jni_LdapOtpNativeInterface.h"

#define NUM_NATIVE_METHODS (3)

extern JPC_CLASS PeerClassArray[];
static char interface_class_name[]           = "gov/llnl/lc/stg/ldapotp/LdapOtpNativeInterface";

static char otp_is_otp_method[]              = "OTP_is_otp";
static char otp_authenticate_method[]        = "OTP_Authenticate";
static char otp_authenticate_pod_method[]    = "OTP_Authenticate";
static char otp_is_otp_signature[]           = "(Ljava/lang/String;)Z";
static char otp_authenticate_signature[]     = "(Ljava/lang/String;Ljava/lang/String;)Z";
static char otp_authenticate_pod_signature[] = "(Ljava/lang/String;Ljava/lang/String;Z)Lgov/llnl/lc/stg/ldapotp/LdapOtpResponse;";

static JNINativeMethod jniNativeMethods[NUM_NATIVE_METHODS];

static JNINativeMethod * otp_is_otp_nm           = &jniNativeMethods[0];
static JNINativeMethod * otp_authenticate_nm     = &jniNativeMethods[1];
static JNINativeMethod * otp_authenticate_pod_nm = &jniNativeMethods[2];

// flag, indicating if this module (and its resources) have been initialized
static int jnu_isInitialized = 0;

// the build date and time, force this to create a "version"
volatile const char version_date[] = __DATE__;
volatile const char version_time[] = __TIME__;

const char* jnu_getLdapOtpNativeVersion(void)
{
  static char VersionString[64];

  sprintf(VersionString, "%s (%s at %s)", LDAPOTP_JNI_VERSION, version_date, version_time);

  return VersionString;
}

int jnu_registerAllNatives(void * pJenv)
{
  JNIEnv * pJEnv = (JNIEnv *) pJenv;
  jclass cls;

  /* initialize the native interface class */
  cls = (*pJEnv)->FindClass(pJEnv, interface_class_name);
  if (cls == NULL)
  {
    fprintf(stderr, "Can't find Class (%s)\n", interface_class_name);
    return -1;
  }

  /* register all the native functions so the jvm can find them */
  otp_is_otp_nm->name = otp_is_otp_method;
  otp_is_otp_nm->signature = otp_is_otp_signature;
  otp_is_otp_nm->fnPtr = Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1is_1otp;

  otp_authenticate_nm->name = otp_authenticate_method;
  otp_authenticate_nm->signature = otp_authenticate_signature;
  otp_authenticate_nm->fnPtr = Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1Authenticate__Ljava_lang_String_2Ljava_lang_String_2;

  otp_authenticate_pod_nm->name = otp_authenticate_pod_method;
  otp_authenticate_pod_nm->signature = otp_authenticate_pod_signature;
  otp_authenticate_pod_nm->fnPtr = Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1Authenticate__Ljava_lang_String_2Ljava_lang_String_2Z;

  (*pJEnv)->RegisterNatives(pJEnv, cls, jniNativeMethods, NUM_NATIVE_METHODS);

  jnu_isInitialized = jpc_initJniReferences(pJEnv);

  return jnu_isInitialized;  // 1 if success
}


void * jnu_getAuthenticateResponse( void* pJenv, const char *username, const char *passcode, unsigned int rtnPod)
{
  static jobject currentObject;
  jstring podStr;
  jstring errStr;
  JNIEnv * pJEnv = (JNIEnv *) pJenv;
  int ldapotpAuthRC = -1;
  char pod[POD_LEN];
  char * p_pod = NULL;

  JPC_CLASS OtpResponse = PeerClassArray[JPC_OTP_RESPONSE_CLASS];
  memset(pod, 0, POD_LEN);

  /* attempt to authenticate the desired way */
  if (rtnPod)
    p_pod = &pod[0];
  ldapotpAuthRC = OTP_Authenticate(username, passcode, p_pod);

  podStr = (*pJEnv)->NewStringUTF(pJEnv, p_pod);
  errStr = (*pJEnv)->NewStringUTF(pJEnv, ldapotp_client_otpError2String(ldapotpAuthRC));

  /* create the response using the constructor */
  currentObject = (*pJEnv)->NewObject(pJEnv, OtpResponse.jpcClass, OtpResponse.constructorMethod->methodID,
      (jboolean) ((ldapotpAuthRC == 0) ? JNI_TRUE : JNI_FALSE), (jint) ldapotpAuthRC, podStr, errStr);

  /* Zero-out the content of the POD buffer */
  memset(pod, 0, POD_LEN);

  if (currentObject == NULL)
    fprintf(stderr, "Error creating the response object");

  return (void *) &currentObject;
}

