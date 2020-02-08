#!/bin/bash

if [ 2 -gt $# ]
then
  FOLDER=$1
else
  FOLDER="../data/*"
fi

COL="$(ls -1A $FOLDER/collision | grep ".csv" | wc -l)"
NOCOL="$(ls -1A $FOLDER/no_collision | grep ".csv" | wc -l)"
echo "   Collisions: $COL"
echo "No collisions: $NOCOL"
echo "        Total: $(($NOCOL+$COL))"
