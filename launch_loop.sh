#!/bin/sh

max=14
for i in `seq 0 $max`
do
    python segment_all_of_sweden.py --use_latest_folder 1 --start_idx "$i" --nbr_outer 15
done
