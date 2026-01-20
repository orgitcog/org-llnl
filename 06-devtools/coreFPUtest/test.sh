#!/bin/sh

corefputestbin=$HOME/corefputest/corefputest
outfile=$HOME/corefputest/$HOSTNAME

mkdir -p $HOME/corefputest
if ! test -e $corefputestbin
then
    echo "Failed to find $corefputest"
    exit 1
fi

date >> $outfile
for i in `numactl --show | grep physcpubind`
do
    if test $i -eq $i 2>/dev/null
    then
        sum=`numactl --physcpubind=$i $corefputestbin --seed=0 --iterations=1000000 --random | md5sum`
        echo $sum | grep c4d45706be3eb2eafe9329a5ab934eec > /dev/null
        if test $? -eq 1
        then
            echo $HOSTNAME: physcpu $i, result = BAD, checksum = $sum >> $outfile
        else
            echo $HOSTNAME: physcpu $i, result = GOOD, checksum = $sum >> $outfile
        fi
    fi
done
