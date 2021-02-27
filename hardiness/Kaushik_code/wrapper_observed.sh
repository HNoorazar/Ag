#!/bin/bash/

for i in {3251..3500..250}
	do
		echo $i
		end=`expr $i + 249`
		qsub hardiness_observed.sh -t $i-$end
		echo "going to sleep"
		sleep 20m
		echo "waking up from sleep"
		echo "Number of file is"$(ls -l /data/hydro/users/kraghavendra/hardiness/output_data/observed/|grep -v ^d|wc -l)	
	done

