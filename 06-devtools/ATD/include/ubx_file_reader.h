#ifndef UBX_FILE_READER_H_
#define UBX_FILE_READER_H_

#include <fstream>
#include "common.h"
#include "ubx_base.h"
#include "ubx_clock.h"
#include "task.h"

// UBX File Reader
//  This unit supports processing the UBX M8 binary protocol (cf. parse_ubx.h)
//  stored in an archive file and accessible via a file stream.  This is a thin
//  wrapper over UbxBase.
//
// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

namespace tmon
{
class UbxFileReader : public UbxBase
{
  public:
    UbxFileReader(string_view name, const ProgState& prog_state, Detector& det,
        string_view filename, std::size_t gnss_idx);
    ~UbxFileReader() override = default;
    UbxFileReader(const UbxFileReader&) = delete;
    UbxFileReader& operator=(const UbxFileReader&) = delete;

  protected:
    void run() override;
    void stop_hook() override;

  private:
    std::ifstream reader_;

    void read_and_handle();
};
} // end namespace tmon

// Copyright (C) 2021, Lawrence Livermore National Security, LLC.
//  All rights reserved. LLNL-CODE-837067
//
// The Department of Homeland Security sponsored the production of this
//  material under DOE Contract Number DE-AC52-07N427344 for the management
//  and operation of Lawrence Livermore National Laboratory. Contract no.
//  DE-AC52-07NA27344 is between the U.S. Department of Energy (DOE) and
//  Lawrence Livermore National Security, LLC (LLNS) for the operation of LLNL.
//  See license for disclaimers, notice of U.S. Government Rights and license
//  terms and conditions.

#endif // UBX_FILE_READER_H_
