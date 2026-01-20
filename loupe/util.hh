//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
//
// Written by Emilio Castillo <ecastill@bsc.es>.
// LLNL-CODE-745958. All rights reserved.
//
// This file is part of Loupe. For details, see:
// https://github.com/LLNL/loupe
// Please also read the LICENSE file for the MIT License notice.
//////////////////////////////////////////////////////////////////////////////

#ifndef __UTIL_HH
#define __UTIL_HH

#include <map>

extern std::map<uint64_t, std::string> g_symbols;

void backtrace(uint64_t *pc_val);
void dump_symbols(std::string* callnames, const std::string& name);

#endif
