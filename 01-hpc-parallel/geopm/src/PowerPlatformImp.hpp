/*
 * Copyright 2017, 2018 Science and Technology Facilities Council (UK)
 * IBM Confidential
 * OCO Source Materials
 * 5747-SM3
 * (c) Copyright IBM Corp. 2017, 2018
 * The source code for this program is not published or otherwise
 * divested of its trade secrets, irrespective of what has
 * been deposited with the U.S. Copyright Office.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY LOG OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef POWERPLATFORMIMP_HPP_INCLUDE
#define POWERPLATFORMIMP_HPP_INCLUDE

#include "PlatformImp.hpp"
#include <nvml.h>

#ifndef NAME_MAX
#define NAME_MAX 1024
#endif

namespace geopm
{
  /// @brief This class provides a base class for Power processor line
  class PowerPlatformImp : public PlatformImp {
  public:
    /// @brief Default constructor.
    PowerPlatformImp();
    /// @brief Copy constructor.
    PowerPlatformImp(const PowerPlatformImp &other);
    /// @brief Default destructor.
    virtual ~PowerPlatformImp();
    
    virtual bool model_supported(int platform_id);
    virtual std::string platform_name(void);
    virtual int power_control_domain(void) const;
    virtual int frequency_control_domain(void) const;
    virtual int performance_counter_domain(void) const;
    virtual void bound(int control_type, double &upper_bound, double &lower_bound);
    virtual double throttle_limit_mhz(void) const;
    virtual double read_signal(int device_type, int device_index, int signal_type);
    virtual void batch_read_signal(std::vector<struct geopm_signal_descriptor> &signal_desc, bool is_changed);
    virtual void write_control(int device_type, int device_index, int signal_type, double value);
    virtual void msr_initialize(void);
    virtual void msr_reset(void);

    virtual bool is_updated(void);

    static int platform_id(void);

  protected:
    const std::string M_MODEL_NAME;
    const int M_PLATFORM_ID;

    std::vector<int> m_occ_file_desc;

    enum {
      M_CLK_UNHALTED_REF,
      M_DATA_FROM_LMEM,
      M_DATA_FROM_RMEM,
      M_INST_RETIRED,
      M_CLK_UNHALTED_CORE
    } m_signal_offset_e;


  private:
    void occ_paths(int chips);
    int occ_open(char* path);
    double occ_read(int idx);

    double cpu_freq_read(int cpuid);

    void pf_event_reset(void);

    /// @brief File path to power-vdd in occ_sensors
    char m_power_path[NAME_MAX];
    /// @brief File path to power-memory in occ_sensors
    char m_memory_path[NAME_MAX];
    // @brief handlers for GPU devices
    // FIXME: it is hard coded for now but information
    // on how many there are should be done in more generic way
    nvmlDevice_t* m_unit_devices_file_desc;
    unsigned int m_total_unit_devices;
    // @brief handlers for CPU frequency
    std::vector<int> m_cpu_freq_file_desc;
  };
}

#endif
