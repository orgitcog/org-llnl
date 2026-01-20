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
 * jni_LdapOtpNativeInterface.c
 *
 *  Created on: Jun 6, 2011
 *      Author: meier3
 */

#include <jni.h>
#include <stdlib.h>
#include <unistd.h>
#include <ldapotp/ldapotp-client.h>
#include <ldapotp/otp_auth_client.h>
#include <ldapotp/ldapotp.h>

#include "jni_LdapOtpNativeInterface.h"
#include "jni_NativeUtils.h"

/*
 *  Require JNI version 1.5 or higher, otherwise the DLL will not load
 *  and will be ...
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
  // put all Native Initialization code here, (refer to initialize for possible fragments)
  JNIEnv * pJenv;
  jint rtnVal = (*vm)->GetEnv(vm, (void **) &pJenv, (jint) JNI_VERSION_1_4);
  if (pJenv != NULL)
  {
    // initialize all references to peer class fields and constructors
    jnu_registerAllNatives((void *) pJenv);
  }
  else
  {
    fprintf(stderr, "Could not obtain a pointer to the java environment (static initializer)\n");
  }

  rtnVal = (jint) JNI_VERSION_1_4;
  return rtnVal;
}

/*
 * Class:     gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface
 * Method:    OTP_is_otp
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1is_1otp
(JNIEnv *pJEnv, jobject jObj, jstring passcode)
{
  const jbyte *strPC;
  jboolean rtn = JNI_FALSE;

  strPC = (*pJEnv)->GetStringUTFChars(pJEnv, passcode, NULL);
  if(strPC != NULL)
  {
    rtn = (jboolean)(OTP_is_otp(strPC) ? JNI_TRUE : JNI_FALSE);
    (*pJEnv)->ReleaseStringUTFChars(pJEnv, passcode, strPC);
  }
  return rtn;
}

/*
 * Class:     gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface
 * Method:    OTP_Authenticate
 * Signature: (Ljava/lang/String;Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1Authenticate__Ljava_lang_String_2Ljava_lang_String_2
(JNIEnv *pJEnv, jobject jObj, jstring username, jstring passcode)
{
  const jbyte *unStr;
  const jbyte *pcStr;
  jboolean rtn = JNI_FALSE;
  ldapotp_int_t ldapotpAuthRC   = -1;

  unStr = (*pJEnv)->GetStringUTFChars(pJEnv, username, NULL);
  pcStr = (*pJEnv)->GetStringUTFChars(pJEnv, passcode, NULL);
  if((unStr != NULL) && (pcStr != NULL))
  {
    ldapotpAuthRC = OTP_Authenticate(unStr, pcStr, NULL);
    rtn = (jboolean)((ldapotpAuthRC == 0) ? JNI_TRUE : JNI_FALSE);
  }
  if(unStr != NULL)
    (*pJEnv)->ReleaseStringUTFChars(pJEnv, username, unStr);
  if(pcStr != NULL)
    (*pJEnv)->ReleaseStringUTFChars(pJEnv, passcode, pcStr);

  return rtn;
}

/*
 * Class:     gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface
 * Method:    OTP_Authenticate
 * Signature: (Ljava/lang/String;Ljava/lang/String;Z)Lgov/llnl/lc/stg/ldapotp/LdapOtpResponse;
 */
JNIEXPORT jobject JNICALL Java_gov_llnl_lc_stg_ldapotp_LdapOtpNativeInterface_OTP_1Authenticate__Ljava_lang_String_2Ljava_lang_String_2Z
(JNIEnv *pJEnv, jobject jObj, jstring username, jstring passcode, jboolean returnPOD)
{
  const jbyte *unStr;
  const jbyte *pcStr;
  void * pObj = NULL;

  unStr = (*pJEnv)->GetStringUTFChars(pJEnv, username, NULL);
  pcStr = (*pJEnv)->GetStringUTFChars(pJEnv, passcode, NULL);
  if((unStr != NULL) && (pcStr != NULL))
  {
    pObj= jnu_getAuthenticateResponse( pJEnv, unStr, pcStr, returnPOD);
  }
  if(unStr != NULL)
    (*pJEnv)->ReleaseStringUTFChars(pJEnv, username, unStr);
  if(pcStr != NULL)
    (*pJEnv)->ReleaseStringUTFChars(pJEnv, passcode, pcStr);

  return  * (jobject *)(pObj);
}


