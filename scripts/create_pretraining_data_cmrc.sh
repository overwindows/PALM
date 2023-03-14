for LANG in train dev trial
do
    python3 -m utils.pre_process_cmrc --input_file /apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus/$LANG.json --output_file /apdcephfs/private_kevinkyhong/data/cmrc2018_public/corpus/$LANG
done
