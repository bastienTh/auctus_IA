#!/bin/bash

if [ 2 -gt $# ]
then
  FOLDER=$1
else
  FOLDER="*"
fi

COL="$(ls -1A ../data/$FOLDER/collision | grep ".csv" | wc -l)"
NOCOL="$(ls -1A ../data/$FOLDER/no_collision | grep ".csv" | wc -l)"
echo "   Collisions: $COL"
echo "No collisions: $NOCOL"
echo "        Total: $(($NOCOL+$COL))"
