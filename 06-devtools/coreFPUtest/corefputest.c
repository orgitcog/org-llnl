/*
Copyright (c) 2010, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Written by Gregory L. Lee <lee218@llnl.gov>.
LLNL-CODE-543371.
All rights reserved.
This file is part of coreFPUtest. For details, see www.llnl.gov. Please also read LICENSE.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  -Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
  -Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
  -Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

void printUsage(int argc, char **argv)
{
    printf("This test is designed to perform a specific calculation that has been shown to exhibit rounding differences on certain \"bad\" AMD Opteron chips.  These differences have been found on model: 65, model name: Dual-Core AMD Opteron(tm) Processor 8216 (information from /proc/cpuinfo).  The resulting output from the test can be piped into md5sum to compare runs across different CPUs.  This test is best run with the numactl utility and the --physcpubind option to bind the process to a specific core.  This test should also be run with the CPU running at full clock rate\n\n");
    printf("USAGE:\n\n");
    printf("\t%s [OPTIONS]\n", argv[0]);
    printf("\tnumactl --physcpubind=<core> %s [OPTIONS]\n\n", argv[0]);
    printf("Options:\n");
    printf("  -h, --help\t\t\tprint this help message and exit\n");
    printf("  -i, --iterations <count>\tperform <count> iterations of the calculation\n");
    printf("  -f, --fixed\t\t\tuse fixed inputs for each iteration\n");
    printf("\t\t\t\t(Note: the fixed input test should NOT be used\n");
    printf("\t\t\t\tas the primary CPU validation test, as the input\n");
    printf("\t\t\t\tis based on one specific \"bad\" CPU)\n");
    printf("  -r, --random\t\t\tuse random inputs for each iteration (default)\n");
    printf("  -s, --seed <value>\t\tseed the random number generator with <value>\n");
    printf("  -b, --brief\t\t\twhen using random inputs, do not print the\n");
    printf("\t\t\t\tresulting calculation for each iteration\n");
    printf("\t\t\t\t(useful when running under a debugger)\n");
    printf("\n");    
}

int main(int argc, char **argv)
{
    double d1, d2, d3, d4, d5;
    double *array;
    unsigned int seed = 0;
    int i, opt, optIndex;
    int iters = 1000000;
    int brief = 0;
    int random = 1;
    struct option opts[] = 
    {
        {"help", no_argument, 0, 'h'},
        {"brief", no_argument, 0, 'b'},
        {"random", no_argument, 0, 'r'},
        {"seed", required_argument, 0, 's'},
        {"iterations", required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    /* Parse the arguments */
    while (1)
    {
        opt = getopt_long(argc, argv, "hbrs:i:", opts, &optIndex);
        if (opt == -1)
            break;
        switch(opt)
        {
            case 'h':
                printUsage(argc, argv);
                exit(0);
            case 'b':
                brief = 1;
                break;
            case 'f':
                random = 0;
                break;
            case 'r':
                random = 1;
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'i':
                iters = atoi(optarg);
                break;
            default:
                break;
        };
    }    

    array = (double *)malloc(iters * sizeof(double));
    srand(seed);

    if (random == 0)
    {
        /* Run with an input set known to generate different answers. 
           Note this input set was derrived from a specific bad CPU and 
           did not exhibit rounding error on another CPU that was found to be
           bad by the random test.  Thus, this particular test is designed
           as a further diagnostic of a CPU that was found to be bad and the
           input values may need to be modified to the specific bad CPU. */
        d1 = 0x000000005e963896;
        d2 = 0x00000000775ba7c1;
        d3 = 0x000000001dd6d6f4;
        d4 = 0x00000000769a091f;
    	
        /* Perform the calculations and save to the array */
        for (i = 0; i < iters; i++)
    	{
            /* The original calculation did negation and addition */
            /* This only reproduced with certain compilers */
            /*d5 = -d1*d2 + d3*d4;*/
            /* With subtraction, this reproduces with more compilers*/
            d5 = d3*d4 - d1*d2;
            array[i] = d5;
    	}
    
        /* Check to see if all calculations are equal */
        for (i = 0; i < iters; i++)
        {
            if (array[i] != array[0])
            {
                printf("TEST FAILED!\n");
                printf("iteration %d: %.16E != %.16E\n", 
                       i, array[i], array[0]);
                return 1;
            }
        }
        printf("TEST PASSED!\n");
        return 0;
    }
    else
    {
        /* Perform the calculations on random numbers */
    	for (i = 0; i < iters; i++)
    	{
            d1 = rand();
            d2 = rand();
            d3 = rand();
            d4 = rand();
            /* The original calculation did negation and addition */
            /* This only reproduced with certain compilers */
            /*d5 = -d1*d2 + d3*d4;*/
            /* With subtraction, this reproduces with more compilers*/
            d5 = d3*d4 - d1*d2;
            array[i] = d5;
            if (brief == 0)
                printf("%lx\n", (long)d5);
    	}
    }

    return 0;
}

