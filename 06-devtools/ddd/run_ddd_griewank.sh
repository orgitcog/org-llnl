#############################################################################################
# Driver script for Delaunay Density Diagnositic (unix version)
#   as used in the paper
#   Algorithm XXXX: The Delaunay Density Diagnostic
#       under review at ACM Transactions on Mathematical Software
#       (original title: Data-driven geometric scale detection via Delaunay interpolation)
#   by Andrew Gillette and Eugene Kur
#   Version 2.0, Jan 2024
#
# This script does the following:
#   1) Run multiple trials of delaunay_density_diagnostic.py for Griewank in dim 2.
#           The outer loop adjusts the scale of inquiry through the "zoom exponent" option
#               but leaves the center of the domain as the origin.
#           The inner loop adjusts the global seed used for randomization.
#           Output consists of one file per trial of the form zz*zoom*seed*.csv
#   2) Save list of all output files into a txt file called allfiles.multi.
#   3) Call generate_ddd_figures.py on allfiles.multi, which does the following: 
#           Generate figure showing MSD and grad-MSD rate as a function 
#               of average sample spacing.  
#           Output figure is displayed and then saved as ddd-figure-griewank.png
#############################################################################################

jobid="123456"
zoomstep="0.4"
maxzoom="4.0"
numtrials="10"

for (( i=0; $i<$(bc<<<"$maxzoom/$zoomstep"); i++ )); do
    for ((j=1; j<$(bc<<<"$numtrials + 1"); j=j+1)) do
        echo
        echo Starting trial with zoom exponent =  $(bc<<<"$zoomstep * $i"), seed = $j
        echo
        python delaunay_density_diagnostic.py --jobid ${jobid} --seed ${j} --zoomexp $(bc<<<"$zoomstep * $i") --numtestperdim 20
    done
done
ls zz-123456*seed*.csv > allfiles.multi
python generate_ddd_figures.py --infile allfiles.multi --outfile ddd-figure-griewank.png --mindens 50 --logscalex