#!/bin/bash

dir=exp
for file in exp1
do
	make ${file}
	# if make sucess
	for i in {1..50}
	do
		echo ${i}
		${dir}/${file}
	done
done
