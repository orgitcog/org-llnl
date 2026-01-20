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
 * jni_PeerClass.h
 *
 *  Created on: Jun 2, 2011
 *      Author: meier3
 */

#ifndef JNI_PEERCLASS_H_
#define JNI_PEERCLASS_H_
#ifdef __cplusplus
extern "C" {
#endif

#define OTP_RESPONSE_CLASS_NAME  "gov/llnl/lc/stg/ldapotp/LdapOtpResponse"
#define STRING_CLASS_NAME        "java/lang/String"

/* only the methods I need to access, listed in static strings above */
#define MAX_NUM_METHODS    (4)            // dont exceed this value
#define NUM_ZERO_METHODS     (0)
#define NUM_STRING_METHODS   (1)

#define MAX_NUM_FIELDS     (4)            // dont exceed this value
#define OTP_RESPONSE_FIELDS (1)
#define NUM_STRING_FIELDS  (0)

#define MAX_STRING_SIZE   (64)

typedef struct JpcMethodID
{
   jmethodID       methodID;  // the method ID
   char *          methodName;
   char *          methodSignature;
} JPC_MID;

typedef struct JpcFieldID
{
   jfieldID        fieldID;  // the field ID
   char *          fieldName;
   char *          fieldSignature;
} JPC_FID;

// wrap the constructors and fields all up in one
typedef struct JpcPeerClass
{
   int             classIndex;
   jclass          jpcClass;
   char*           className;          //
   JPC_MID *       constructorMethod;
   JPC_MID *       methodArray;
   JPC_FID *       fieldArray;
   int             numMethods;
   int             numFields;
} JPC_CLASS;


// indexes for the peer class array  (keep this order)
enum JPC_PEER_CLASS_TYPE
{
  JPC_OTP_RESPONSE_CLASS = 0,
  JPC_STRING_CLASS,
  JPC_NUM_PEER_CLASSES    // always last, this enum defines the order!!
};

int jpc_initJniReferences(void * pJenv);

#ifdef __cplusplus
}
#endif
#endif /* JNI_PEERCLASS_H_ */
