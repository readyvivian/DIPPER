SOURCE=/data/swalia/dipper/DIPPER_data/AliSim/dataset/10K/data.unaligned.fa.gz
GT=/home/v8wang@AD.UCSD.EDU/DIPPER/output/data.treefile.nwk

MAPLE=/home/swalia@AD.UCSD.EDU/tools/MAPLE/MAPLEv0.3.6.py
DIPPER=/home/v8wang@AD.UCSD.EDU/DIPPER/bin/dipper

OUTPUT_DIR=/home/v8wang@AD.UCSD.EDU/DIPPER/output
OUTPUT_NAME=tree_pl.nwk


$DIPPER -i r -I $SOURCE -O $OUTPUT_DIR/$OUTPUT_NAME -m 1

python3 $MAPLE --inputTree $GT --inputRFtrees $OUTPUT_DIR/$OUTPUT_NAME --output $OUTPUT_DIR/maple_out --overwrite

awk 'NR==2 {print $2}' $OUTPUT_DIR/maple_out_RFdistances.txt
