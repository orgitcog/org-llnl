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

#include "geopm_arch.h"
#include "PowerPlatformImp.hpp"
#include "geopm_message.h"
#include "geopm_error.h"
#include "Exception.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <fcntl.h>

/* FIXME: This is hard coded for now, but
 * we should see whether there is a way
 * to find programatically out number of hyperthreads 
 * once SMT is enabled */
#define HYPERTHREADS 8

namespace geopm
{

  static const std::map<std::string, std::pair<off_t, unsigned long> > &power8_hwc_map(void);

  int PowerPlatformImp::platform_id(void) {
    /* FIXME: Randomly picked number
     * some better id should be used */
    return 12;
  }

  PowerPlatformImp::PowerPlatformImp() 
    : PlatformImp(2, 6, 50.0, &(power8_hwc_map()))
    , M_MODEL_NAME("Power8")
    , M_PLATFORM_ID(platform_id())
    , m_total_unit_devices(0) {
  }

  PowerPlatformImp::PowerPlatformImp(const PowerPlatformImp &other) 
    : PlatformImp(other)
    , M_MODEL_NAME(other.M_MODEL_NAME)
    , M_PLATFORM_ID(other.M_PLATFORM_ID)
    , m_total_unit_devices(0) {
    }

  PowerPlatformImp::~PowerPlatformImp() {
    for(int c = 0; c < m_num_logical_cpu; ++c) {
      free(m_pf_event_read_data[c]);
    }
    free(m_pf_event_read_data);

    for(std::vector<int>::iterator it = m_occ_file_desc.begin();
	it != m_occ_file_desc.end();
	++it) {
      close(*it);
      *it = -1;
    }

    for(std::vector<int>::iterator it = m_cpu_freq_file_desc.begin();
	it != m_cpu_freq_file_desc.end();
	++it) {
      close(*it);
      *it = -1;
    }

    free(m_unit_devices_file_desc);
      
    /* shutdown */
    nvmlShutdown();
  }

  bool PowerPlatformImp::model_supported(int platform_id) {
    return (M_PLATFORM_ID == platform_id);
  }

  std::string PowerPlatformImp::platform_name() {
    return M_MODEL_NAME;
  }

  int PowerPlatformImp::power_control_domain(void) const {
    return GEOPM_DOMAIN_PACKAGE;
  }

  int PowerPlatformImp::frequency_control_domain(void) const {
    return GEOPM_DOMAIN_PACKAGE_CORE;
  }

  int PowerPlatformImp::performance_counter_domain(void) const {
    return GEOPM_DOMAIN_PACKAGE;
  }

  void PowerPlatformImp::bound(int control_type, double &upper_bound, double &lower_bound) {
    upper_bound = 1000;
    lower_bound = 0;
  }

  double PowerPlatformImp::throttle_limit_mhz(void) const {
    return 0.5; // the same value as it is in KNL implementation
  }

  double PowerPlatformImp::read_signal(int device_type, int device_index, int signal_type) {
    double value = 0.0;
    int offset_idx = 0;

    switch(signal_type) {
      case GEOPM_TELEMETRY_TYPE_PKG_ENERGY:
	offset_idx = device_index * m_num_energy_signal;
	value = occ_read(offset_idx + 1);
	break;

      case GEOPM_TELEMETRY_TYPE_DRAM_ENERGY:
	offset_idx = device_index * m_num_energy_signal + 1;
	value = occ_read(offset_idx + 1);
	break;

      case GEOPM_TELEMETRY_TYPE_FREQUENCY:
	{
	  value = 0.0;

	  int cpu_per_socket = m_num_logical_cpu / m_num_package;
	  for(int cpu = device_index * cpu_per_socket;
	      cpu < (device_index + 1) * cpu_per_socket;
	      ++cpu) {
	    value += cpu_freq_read(cpu);
	  }

	  value /= (double)cpu_per_socket;

	  break;
	}

      case GEOPM_TELEMETRY_TYPE_INST_RETIRED:
	{
	  value = 0;

	  int cpu_per_socket = m_num_logical_cpu / m_num_package;
	  for(int cpu = device_index * cpu_per_socket;
	      cpu < (device_index + 1) * cpu_per_socket;
	      ++cpu) {
	    value += m_pf_event_read_data[cpu][M_INST_RETIRED + 1];
	  }

	  break;
	}

      case GEOPM_TELEMETRY_TYPE_CLK_UNHALTED_CORE: 
	{
	  value = 0;

	  int cpu_per_socket = m_num_logical_cpu / m_num_package;
	  for(int cpu = device_index * cpu_per_socket;
	      cpu < (device_index + 1) * cpu_per_socket;
	      ++cpu) {
	    
	    value += m_pf_event_read_data[cpu][M_CLK_UNHALTED_CORE + 1];
	  }
	  
	  break;
	}

      case GEOPM_TELEMETRY_TYPE_CLK_UNHALTED_REF:
	{
	  value = 0;

	  int cpu_per_socket = m_num_logical_cpu / m_num_package;
	  for(int cpu = device_index * cpu_per_socket;
	      cpu < (device_index + 1) * cpu_per_socket;
	      ++cpu) {
	    
	    value += m_pf_event_read_data[cpu][M_CLK_UNHALTED_REF + 1];
	  }

	  break;
	}

      case GEOPM_TELEMETRY_TYPE_READ_BANDWIDTH:
	{
	  value = 0;
	  int cpu_per_socket = m_num_logical_cpu / m_num_package;
	  for(int cpu = device_index * cpu_per_socket;
	      cpu < (device_index + 1) * cpu_per_socket;
	      ++cpu) {
	    value += m_pf_event_read_data[cpu][M_DATA_FROM_LMEM + 1] +
	      m_pf_event_read_data[cpu][M_DATA_FROM_RMEM + 1];
	  }

	  break;
	}

      case GEOPM_TELEMETRY_TYPE_GPU_ENERGY:
	{
	  /* FIXME: The assumption at the moment is 
	   * that number of GPUs is evenly spread between sockets,
	   * but we should look at finding that out
	   * in some programmatic way */
	  value = 0.0;
	  
	  unsigned int power;
	  int gpus_per_socket = m_total_unit_devices / m_num_package;
	  for(int d = device_index * gpus_per_socket;
	      d < (device_index + 1) * gpus_per_socket;
	      ++d) {
	    nvmlDeviceGetPowerUsage(m_unit_devices_file_desc[d], &power);
	    value += (double)power * 0.001;
	  }

	  break;
	}
	  
      default:
	throw geopm::Exception("PowerPlatformImp::read_signal: Invalid signal type", 
			       GEOPM_ERROR_INVALID, 
			       __FILE__, 
			       __LINE__);
                break;
    }

    return value;
  }

  void PowerPlatformImp::batch_read_signal(std::vector<struct geopm_signal_descriptor> &signal_desc, bool is_changed) {
    // TODO: is there a way we can read all signals in a batch on Power architecture?
    // Using at te moment serial code path

    /* Obtain results from performance counters */
    for(int cpu = 0; cpu < m_num_logical_cpu; ++cpu)
      pf_event_read(cpu);

    for(auto it = signal_desc.begin(); it != signal_desc.end(); ++it) {
      (*it).value = read_signal((*it).device_type, (*it).device_index, (*it).signal_type);
    }

  }

  void PowerPlatformImp::write_control(int device_type, int device_index, int signal_type, double value) {
    // Do nothing for now

    // TODO
  }

  void PowerPlatformImp::occ_paths(int chip) {
    struct stat s;
    int err;

    snprintf(m_power_path, NAME_MAX, "/sys/devices/system/cpu/occ_sensors/chip%d/power-vdd", chip);
    err = stat(m_power_path, &s);
    if(err == 0) {
      snprintf(m_memory_path, NAME_MAX, "/sys/devices/system/cpu/occ_sensors/chip%d/power-memory", chip);
      err = stat(m_memory_path, &s);
      if(err == 0) 
	return;
    }
    
    throw Exception("no power-vdd or power-memory in occ_sensors directory", 
		    GEOPM_ERROR_MSR_OPEN,
		    __FILE__,
		    __LINE__);
  }

  int PowerPlatformImp::occ_open(char* path) {
    int fd;

      fd = open(path, O_RDONLY);
      //report errors
      if (fd < 0) {
	char error_string[NAME_MAX];
	if (errno == ENXIO || errno == ENOENT) {
	  snprintf(error_string, NAME_MAX, "device %s does not exist", path);
	}
	else if (errno == EPERM || errno == EACCES) {
	  snprintf(error_string, NAME_MAX, "permission denied opening device %s", path);
	}
	else {
	  snprintf(error_string, NAME_MAX, "system error opening cpu device %s", path);
	}
	throw Exception(error_string, GEOPM_ERROR_MSR_OPEN, __FILE__, __LINE__);
	
	return -1;
      }

      return fd;
  }

  double PowerPlatformImp::occ_read(int idx) {
    double value = 0.0;

    char file_text[BUFSIZ];

    int rv = pread(m_occ_file_desc[idx], &file_text[0], sizeof(file_text), 0);
    if(rv <= 0) {
      throw Exception("no file descriptor found for OCC device", 
		      GEOPM_ERROR_MSR_READ, 
		      __FILE__,
		      __LINE__);
    }

    std::string s(file_text);
    std::string watts = s.substr(0, s.find(' '));
    
    value = atof(watts.c_str());
    
    return value;
  }

  double PowerPlatformImp::cpu_freq_read(int cpuid) {
    double value = 0.0;

    char file_text[BUFSIZ];
    pread(m_cpu_freq_file_desc[cpuid], &file_text[0], sizeof(file_text), 0);
    
    std::string s(file_text);
    value = atof(s.c_str()) / 1e6; /* in GHz */
      
    return value;
  }

  bool PowerPlatformImp::is_updated(void) {
    uint64_t curr_value = (uint64_t)occ_read(0);
    bool result = (m_trigger_value && curr_value != m_trigger_value);
    m_trigger_value = curr_value;

    return result;
  }

  void PowerPlatformImp::msr_initialize() {
    int fd_energy, fd_power, fd_memory;
    struct stat s;

    /* Frequency information */
    char cpu_freq_path[NAME_MAX];

    /* TODO: Fix to something more
     * dynamic then hard coded values */
    int step = 1;
    if(m_num_cpu_per_core == 1)
      step = HYPERTHREADS;
    for(int c = 0; c < m_num_logical_cpu; c++) {
      struct stat s;
      int fd = -1;

      snprintf(cpu_freq_path, NAME_MAX, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", c*step);
      int err = stat(cpu_freq_path, &s);
      if(err == 0) 
	fd = open(cpu_freq_path, O_RDONLY);

      m_cpu_freq_file_desc.push_back(fd);
    }

    char energy_path[NAME_MAX];
    snprintf(energy_path, NAME_MAX, "/sys/devices/system/cpu/occ_sensors/chip0/chip-energy");
    int err = stat(energy_path, &s);
    if(err != 0) 
      throw Exception("no chip-energy in occ_sensors directory", 
		      GEOPM_ERROR_MSR_OPEN,
		      __FILE__,
		      __LINE__);

    fd_energy = occ_open(energy_path);
    assert(fd_energy > 0);
    m_occ_file_desc.push_back(fd_energy);

    int energy_domains = num_domain(power_control_domain());
    for(int i = 0; i < energy_domains; ++i) {
      occ_paths(i);
      fd_power = occ_open(m_power_path);
      assert(fd_power > 0);
      m_occ_file_desc.push_back(fd_power);

      fd_memory = occ_open(m_memory_path);
      assert(fd_memory > 0);
      m_occ_file_desc.push_back(fd_memory);
    }

    /* Initialize GPU reading */
    nvmlReturn_t result;
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
      /* TODO: gracefully fail if there is no GPUs 
       * on the platform */
      throw Exception("PowerPlatformImp::initialize: Failed to initialize NVML\n",
		      GEOPM_ERROR_RUNTIME, 
		      __FILE__, 
		      __LINE__);
    }

    nvmlDeviceGetCount(&m_total_unit_devices);
    m_unit_devices_file_desc = (nvmlDevice_t*) malloc(sizeof(nvmlDevice_t) * m_total_unit_devices);

    /* get handles to all devices */
    for(unsigned int d = 0; d < m_total_unit_devices; ++d) {
      int power;
      char msg[128];

      result = nvmlDeviceGetHandleByIndex(d, &m_unit_devices_file_desc[d]);
      if (result != NVML_SUCCESS) {
	sprintf(msg, "PowerPlatformImp::initialize: Failed to get handle for device %d: %s\n", d, nvmlErrorString(result));
	throw Exception(msg,
		      GEOPM_ERROR_RUNTIME, 
		      __FILE__, 
		      __LINE__);
      }

      /* check to see whether we can read power */
      result = nvmlDeviceGetPowerUsage(m_unit_devices_file_desc[d], (unsigned int *)&power);
      if (result != NVML_SUCCESS) {
	sprintf(msg, "PowerPlatformImp::initialize: Failed to read power on device %d: %s\n", d, nvmlErrorString(result));
	throw Exception(msg,
			GEOPM_ERROR_RUNTIME, 
			__FILE__, 
			__LINE__);
      }
    }

    m_pf_event_read_data = (uint64_t**)malloc(m_num_logical_cpu * sizeof(uint64_t*));

    for(int c = 0; c < m_num_logical_cpu; c++) {
      pf_event_open(c*step);
      m_pf_event_read_data[c] = (uint64_t*) malloc(pf_event_read_data_size());
    }


    /* Enable all hardware counters */
    for(int c = 0; c < m_num_logical_cpu; ++c) {
      ioctl(m_cpu_file_desc[c], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
  }

  void PowerPlatformImp::pf_event_reset() {
    for(int i = 0; i < m_num_logical_cpu; ++i) {
      ioctl(m_cpu_file_desc[i], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    }
  }

  void PowerPlatformImp::msr_reset() {
    pf_event_reset();
  }

  static const std::map<std::string, std::pair<off_t, unsigned long> > &power8_hwc_map(void) {
    static const std::map<std::string, std::pair<off_t, unsigned long> > r_map({
	{ "PM_CYC",            {0x1e,    0x00}},
	{ "PM_DATA_FROM_LMEM", {0x2c048, 0x0}},
	{ "PM_DATA_FROM_RMEM", {0x3c04a, 0x0}},
	{ "PM_RUN_INST_CMPL",  {0x500fa, 0x0}},
	{ "PM_RUN_CYC",        {0x600f4, 0x0}}
      });
    
    return r_map;
  }

  
}
