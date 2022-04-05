#set -eu 
#set -o pipefail 


# Attention! Python 2.7.14  and python3 gives different vocabulary order. We use Python 2.7.14 to preprocess files.

# input files: train.txt valid.txt test.txt  
# (these are default filenames, change files name with the following arguments:  --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt
python3 ./kbc_data_preprocess.py --task fb15k --dir ./fb15k
python3 ./kbc_data_preprocess.py --task wn18 --dir ./wn18
python3 ./kbc_data_preprocess.py --task fb15k237 --dir ./fb15k237
python3 ./kbc_data_preprocess.py --task wn18rr --dir ./wn18rr

# input files: train dev test
# (these are default filenames, change files name with the following arguments: --train $trainname --valid $validname --test $testname)
# output files: vocab.txt train.coke.txt valid.coke.txt test.coke.txt sen_candli.txt trivial_sen.txt
python3 ./pathquery_data_preprocess.py --task pathqueryFB --dir ./pathqueryFB 
python3 ./pathquery_data_preprocess.py --task pathqueryWN --dir ./pathqueryWN
