#############################################################################################
# Test script for Delaunay Density Diagnositic
#   based on the paper
#   Data-driven geometric scale detection via Delaunay interpolation
#   by Andrew Gillette and Eugene Kur
#   Version 1.0, May 2023
#
# This script does a dry run of one iteration of the main inner loop
#   of the script run_ddd_trials.sh.  If this script succeeds, the
#   script run_ddd_trials.sh should be able to run without error.
#############################################################################################

jobid="999999"
zoomstep="4.0"
maxzoom="4.0"
numtrials="1"

for (( i=0; $i<$(bc<<<"$maxzoom/$zoomstep"); i++ )); do
    for ((j=1; j<$(bc<<<"$numtrials + 1"); j=j+1)) do
        echo
        echo Starting trial with zoom exponent =  $(bc<<<"$zoomstep * $i"), seed = $j
        echo
        python delaunay_density_diagnostic.py --jobid ${jobid} --seed ${j} --zoomexp $(bc<<<"$zoomstep * $i") --numtestperdim 20
    done
done
rm zz-999999*.csv 
echo ""
echo "    (test_install.sh deleted temporary file zz-999999*.csv)"
echo ""
echo "*** If no errors above, the test ran sucessfully. ***"
echo "*** Try running the script run_ddd_trials.sh next. ***"