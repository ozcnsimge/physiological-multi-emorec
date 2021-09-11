#!/bin/bash

for zipfile in *.zip; do
	IFS=$'\n'
	targets=(`unzip -l "${zipfile}" | grep -oP T.+.FACS$`)
	for facs in ${targets[@]}; do
		unslashed=$(echo ${facs////-})
		name=$(echo ${unslashed// /-})
		unzip -p "$zipfile" $facs > "$name".xlsx
	done
done
