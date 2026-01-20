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
 * jni_PeerClass.c
 *
 *  Created on: Jun 2, 2011
 *      Author: meier3
 */
#include <jni.h>

#include "jni_PeerClass.h"

 JPC_CLASS PeerClassArray[JPC_NUM_PEER_CLASSES];
 JPC_FID   FieldIdArray[JPC_NUM_PEER_CLASSES][MAX_NUM_FIELDS];
 JPC_MID   MethodIdArray[JPC_NUM_PEER_CLASSES][MAX_NUM_METHODS];

/* these are the peer classes, and this ORDER must be maintained for all below */
static char *java_peer_class_names[] =
{
    OTP_RESPONSE_CLASS_NAME,
    STRING_CLASS_NAME,
};


static int methods_in_class[] =
{
    NUM_ZERO_METHODS,
    NUM_STRING_METHODS,
};

static int fields_in_class[] =
{
    OTP_RESPONSE_FIELDS,
    NUM_STRING_FIELDS,
};

static char constructor_method_name[] = "<init>";  // default - class name

static char *constructor_method_signatures[] =
{
  "(ZILjava/lang/String;Ljava/lang/String;)V",            // OTP Response
  "()V",                                                  // string
};

/* TODO finish these, for now just implement the constructors */
static char * method_names[][MAX_STRING_SIZE] =
{
  {
      constructor_method_name,
  },
  {
      constructor_method_name,
  },
};
static char *method_signatures[][MAX_STRING_SIZE] =
{
  {
      "()V",
  },
  {
      "()V",
  },
};


static char * field_names[][MAX_STRING_SIZE] =
{
  {
    "authenticated",                    // OTP_Response
    "returnCode",
    "Pod",
    "ErrorMsg",
  },
  {
    "stringID",
  },
};
static char *field_signatures[][64] =
{
  {
      "Z",                    // OTP_Response
      "I",
      "Ljava/lang/String;",
      "Ljava/lang/String;",
  },
  {
     "J",
  },
};

/************************************************************************
*** Function: jpc_initFieldID
***
*** This is a JNI optimization.  Finds the references to a known field in a
*** class, and caches it for later use.
***
*** Called by jpc_initAllFields()
*** <p>
***
*** created:  9/22/2003 (3:05:33 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initFieldID(JNIEnv * pJEnv, jclass jClass, JPC_FID * fidStruct)
{
  int success = 0;
  jfieldID tempFID = NULL;

  tempFID = (*pJEnv)->GetFieldID(pJEnv, jClass, fidStruct->fieldName , fidStruct->fieldSignature);

  if(tempFID == NULL)
  {
    fprintf(stderr,"JNI cannot create a field Id for (%s)!\n", fidStruct->fieldName);
  }
  else
  {
    fidStruct->fieldID = tempFID;
    success = 1;
  }
  return success;
}
/*-----------------------------------------------------------------------*/


/******************************************************************************
*** Function: jpc_initAllFields
***
*** This is a JNI optimization.  Finds the references to the known fields in a
*** class, and caches them for later use.  Primarily used by the default
*** constructor of a class.
***
*** Called by jpc_initPeerClass()
*** <p>
***
*** created:  9/22/2003 (3:05:33 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initAllFields(JNIEnv * pJEnv, JPC_CLASS * classStruct)
{
  int success                     = 1;
  int index                       = classStruct->classIndex;
  int j;
  int numFields                   = classStruct->numFields;
  jclass jClass                   = classStruct->jpcClass;

  JPC_FID *pFID;

  for(j = 0; j <  numFields; j++)
  {
    pFID = &(classStruct->fieldArray[j]);
    pFID->fieldName      = field_names[index][j];
    pFID->fieldSignature = field_signatures[index][j];
    if( !jpc_initFieldID( pJEnv, jClass, pFID) )
    {
      fprintf(stderr,"JNI cannot create a field Id for (%s)!\n", pFID->fieldName);
      success = 0;
    }
  }
  return success;
}
/*-----------------------------------------------------------------------*/


/******************************************************************************
*** Function: jpc_initConstructorID
***
*** This is a JNI optimization.  Finds the references to the known methods in a
*** class, and caches them for later use.
***
*** Called by jpc_initPeerClass()
*** <p>
***
*** created:  9/22/2003 (5:04:23 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initConstructorID(JNIEnv * pJEnv, jclass jClass, JPC_MID * midStruct)
{
  int success = 0;
  jmethodID tempMID = NULL;

  tempMID = (*pJEnv)->GetMethodID(pJEnv, jClass, midStruct->methodName , midStruct->methodSignature);

  if(tempMID == NULL)
  {
    fprintf(stderr,"JNI cannot create a method Id for (%s) {%s}!\n", midStruct->methodName, midStruct->methodSignature);
  }
  else
  {
    midStruct->methodID = tempMID;
    success = 1;
  }
  return success;
}
/*-----------------------------------------------------------------------*/

/******************************************************************************
*** Function: jpc_initMethodID
***
*** This is a JNI optimization.  Finds the references to the known methods in a
*** class, and caches them for later use.
***
*** Called by jpc_initPeerClass()
*** <p>
***
*** created:  9/22/2003 (5:04:23 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initMethodID(JNIEnv * pJEnv, jclass jClass, JPC_MID * midStruct)
{
  int success = 0;
  jmethodID tempMID = NULL;

  tempMID = (*pJEnv)->GetMethodID(pJEnv, jClass, midStruct->methodName , midStruct->methodSignature);

  if(tempMID == NULL)
  {
    fprintf(stderr,"JNI cannot create a method Id for (%s)!\n", midStruct->methodName);
  }
  else
  {
    midStruct->methodID = tempMID;
    success = 1;
  }
  return success;
}
/*-----------------------------------------------------------------------*/

/******************************************************************************
*** Function: jpc_initAllMethods
***
*** This is a JNI optimization.  Finds the references to the known methods in a
*** class, and caches them for later use.
***
*** Called by jpc_initPeerClass()
*** <p>
***
*** created:  9/22/2003 (3:05:33 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initAllMethods(JNIEnv * pJEnv, JPC_CLASS * classStruct)
{
  int success                     = 1;
  int index                       = classStruct->classIndex;
  int j;
  int numMethods                  = classStruct->numMethods;
  jclass jClass                   = classStruct->jpcClass;

  JPC_MID *pMID;

  for(j = 0; j <  numMethods; j++)
  {
    pMID = &(classStruct->methodArray[j]);
    pMID->methodName      = method_names[index][j];
    pMID->methodSignature = method_signatures[index][j];
    if( !jpc_initMethodID( pJEnv, jClass, pMID) )
    {
      fprintf(stderr,"JNI cannot create a method Id for (%s)!\n", pMID->methodName);
      success = 0;
    }
  }
  return success;
}
/*-----------------------------------------------------------------------*/



/******************************************************************************
*** Function: jpc_initPeerClasses
***
*** This is a JNI optimization.  Finds the references to the known classes that
*** will be used within C, and cache them for later use.
***
*** Called by jpc_initJniReferences()
*** <p>
***
*** created:  9/22/2003 (4:46:46 PM)
***
***   Paramters:
***
***   Returns:
***
******************************************************************************/
int jpc_initPeerClass(JNIEnv * pJEnv, JPC_CLASS * classStruct)
{
  int success = 0;
  int index   = classStruct->classIndex;

  jclass localClassRef;

  classStruct->className = java_peer_class_names[index];

  localClassRef = (*pJEnv)->FindClass(pJEnv, classStruct->className );
  if(localClassRef == NULL)
  {
    fprintf(stderr,"JNI cannot create an local reference for (%s)!\n", classStruct->className);
  }
  else
  {
    classStruct->jpcClass = (*pJEnv)->NewGlobalRef(pJEnv, localClassRef);
    if(classStruct->jpcClass == NULL)
    {
      fprintf(stderr,"JNI cannot create an global class reference for (%s)!\n", classStruct->className);
    }
    else
    {
      // success so far, now get a reference to the constructor
      classStruct->constructorMethod->methodName      = constructor_method_name;
      classStruct->constructorMethod->methodSignature = constructor_method_signatures[index];
      if(jpc_initConstructorID( pJEnv, classStruct->jpcClass, classStruct->constructorMethod ))
      {

        success = jpc_initAllMethods(pJEnv, classStruct);

        // finally, fill in all the field ids
        success = jpc_initAllFields(pJEnv, classStruct);
      }
    }
    // get rid of the local references
    (*pJEnv)->DeleteLocalRef(pJEnv, localClassRef);
  }
  return success;
}
/*-----------------------------------------------------------------------*/


/**************************************************************************
*** Method Name:
***     jpc_initJniReferences
**/
/**
*** This is a JNI optimization.  Find all the references to the known data
*** structures and methods, and cache them for later use.  This causes the
*** initialization phase (one-time event) to be a bit slower, and consume
*** memory, with the benefit of much faster execution times.
*** <p>
***
*** @see          Method_related_to_this_method
***
*** @param        Parameter_name  Description_of_method_parameter__Delete_if_none
***
*** @return       Description_of_method_return_value__Delete_if_none
***
*** @throws       Class_name  Description_of_exception_thrown__Delete_if_none
**************************************************************************/

int jpc_initJniReferences(void * pJenv)
{
  int success = 1;
  int numClasses = JPC_STRING_CLASS;  // JPC_NUM_PEER_CLASSES
  int j;
  JNIEnv * pJEnv = (JNIEnv *)pJenv;

  for(j = 0; j < numClasses; j++)
  {
    // initialize as much of the class as possible
    PeerClassArray[j].classIndex        = j;
    PeerClassArray[j].className         = java_peer_class_names[j];
    PeerClassArray[j].constructorMethod = &(MethodIdArray[j][0]);
    PeerClassArray[j].methodArray       = &(MethodIdArray[j][0]);
    PeerClassArray[j].fieldArray        = &(FieldIdArray[j][0]);
    PeerClassArray[j].numMethods        = methods_in_class[j];
    PeerClassArray[j].numFields         = fields_in_class[j];

    // and then go get the references for fields and methods in each class
    //  *** note:  if a class depends upon another class, then they need to be
    //  *** initialized in order.  Preserve the original array order, dependencies
    //  *** are satisfied!

    if(!jpc_initPeerClass(pJEnv, &PeerClassArray[j]) )
    {
      fprintf(stderr,"JNI cannot create an global reference for (%s)!\n", PeerClassArray[j].className);
      success = 0;
    }
  }
  return success;
}
/*-----------------------------------------------------------------------*/
