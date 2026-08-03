#!/bin/bash
L="$1"
echo "  ready to roll        : $(strings $L | grep -c 'ready to roll')"
echo "  Memory access fault  : $(strings $L | grep -c 'Memory access fault')"
echo "  HSA OUT_OF_RESOURCES : $(strings $L | grep -c 'HSA_STATUS_ERROR_OUT_OF_RESOURCES')"
echo "  Fatal Python error   : $(strings $L | grep -c 'Fatal Python error')"
echo "  Traceback            : $(strings $L | grep -c 'Traceback')"
echo "  Scheduler exception  : $(strings $L | grep -c 'Scheduler hit an exception')"
echo "  Expected lengths.size: $(strings $L | grep -c 'Expected lengths.size')"
echo "  kvd adapter connected: $(strings $L | grep -c 'infera-kvd adapter connected')"
echo "  AiterCustomAllreduce : $(strings $L | grep -c 'AiterCustomAllreduce')"
echo "  AR path NCCL         : $(strings $L | grep -c 'All-reduce call path: NCCL')"
echo "  MC_FORCE_TCP         : $(strings $L | grep -c 'MC_FORCE_TCP')"
strings $L | grep -m1 -oE 'mem_fraction_static=[0-9.]+'
strings $L | grep -m1 -oE 'context_length=[0-9]+'
strings $L | grep -m1 -oE "speculative_algorithm='?[A-Za-z]+'?|speculative_algorithm=None"
strings $L | grep -m1 -oE 'disable_custom_all_reduce=[A-Za-z]+'
echo "  Memory pool end      : $(strings $L | grep -m1 -oE 'avail mem=[0-9.]+ GB')"
echo "  max_total_num_tokens : $(strings $L | grep -m1 -oE 'max_total_num_tokens=[0-9]+')"
echo "  Errno 98 after ready : $(strings $L | sed -n '/ready to roll/,$p' | grep -c 'Errno 98')"
