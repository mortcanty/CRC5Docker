#!/usr/bin/bash
#
# Usage: 
# run_sar_seqQ.sh pattern imdir enl significance 
# // in last line means global replacement of all occurrences

alpha="${@: -1}"
enl=("${@: -2}")
imdir=("${@: -3}")

fns=$(ls -l $imdir | grep $1 | \
     grep -v 'sarseq' | grep -v 'enl' | \
     grep -v 'mmse' | grep -v 'gamma' | \
     grep -v '.hdr' | \
     grep -v 'warp' | grep -v 'sub' |  awk '{print $9}')
     
python3 scripts/sar_seqQ.py -m -s  $alpha  \
                     ${fns//$1/$imdir$1} sarseqQ.tif $enl 