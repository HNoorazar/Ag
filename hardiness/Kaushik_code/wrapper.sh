#!/bin/bash/

for i in {84501..90128..250}
	do
		echo $i
		end=`expr $i + 249`
		echo "Going to sleep"
		qsub hardiness.sh -t $i-$end
		sleep 30m
		echo "Waking up from sleep"
		echo "Number of files is"
		echo $(ls -l /data/hydro/users/kraghavendra/hardiness/output_data/CanESM2/rcp85/ | grep -v ^d |wc -l)
		
	done

