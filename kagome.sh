#!/bin/sh

Hz=2.30
end=2.50 
step=0.01

result=`echo "$Hz < $end" | bc`

while [ $result -eq 1 ]
do
	python3 SU-kagome.py 6 $Hz 0.1
	Hz=`echo "scale=3; $Hz + $step" | bc`
	result=`echo "$Hz < $end" | bc`
done