num_gpu=$1
horovodrun -np $num_gpu -H localhost:$num_gpu python3 -u ../../train.py -c grad_tts_blank.json -l ../../logdir -m gradtts_n_1000