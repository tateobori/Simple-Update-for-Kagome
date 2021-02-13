#!/bin/sh

Hz=2.0
end=4.0 
step=0.05

result=`echo "$Hz < $end" | bc`

while [ $result -eq 1 ]
do
	python3 SU-husimi.py 4 $Hz 0.01
	Hz=`echo "scale=3; $Hz + $step" | bc`
	result=`echo "$Hz < $end" | bc`
done