## list of commands
****

- check_node
- copy
- crc32
- env
- flush_file
- get_jobstep_id
- glob_hosts
- halt
- halt_cntl
- index
- inspect
- inspect_cache
- kill_jobstep
- list_dir
- list_down_nodes
- log_event
- log_transfer
- nodes_file
- postrun
- prefix
- prerun
- print_hash_file
- rebuild_xor
- retries_halt
- scavenge
- test_datemanip
- test_runtime
- transfer
- watchdog

## list of launchers
****

- srun
- jsrun
- aprun
- mpirun

debug set to False by default
if SCR_DEBUG > 0, then set debug to True

**line 117**: `'/scr_' `+ args.command

**line 144**: `scr_test_runtime debug rc`

- (scr/scripts/common)

**line 148**: `scr_env --jobid debug`
	
- scr/scripts/LSF
- scr/scripts/PMIX
- scr/scripts/TLCC
- scr/scripts/cray_xt

**line 159**: `scr_list_dir control debug both`
	
- scr/scripts/common

**line 171**: `scr_prerun -p prefix debug rc`

- scr/scripts/common

**line 179**: `scr_glob_hosts --count --hosts scr_nodelist debug output`

- scr/scripts/common

**line 213**: `scr_list_down_nodes {--free} debug output`

- scr/scripts/LSF
- scr/scripts/PMIX
- scr/scripts/TLCC
- scr/scripts/cray_xt

**if there are down nodes:**

**line 217**:	`scr_list_down_nodes {-free} --log --reason --secs 0 debug output none`

**line 229**: `scr_glob_hosts --count --hosts scr_nodelist debug output`

- scr/scripts/common

**line 236**: `scr_env --prefix prefix --runnodes debug both`

- scr/scripts/LSF
- scr/scripts/PMIX
- scr/scripts/TLCC
- scr/scripts/cray_xt

**line 243**: `scr_glob_hosts --count --minus scr_nodelist down_nodes`
		
- scr/scripts/common

****

**line 275**: `scr_log_event -T RUN_STARTED -N Job={jobid} Run={attempts} -S start_secs debug none`

- src/

**line 293**: `scr_get_jobstep_id scr_p.pid debug output`

- scr/scripts/LSF
- scr/scripts/PMIX
- scr/scripts/TLCC
- scr/scripts/cray_xt
