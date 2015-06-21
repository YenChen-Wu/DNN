#dir=/media/wyc/A4F4F960F4F93560/nn_post
dir=~/local_home/nn_post

#./svm_hmm_classify -v 0 --n 2000 $dir/test.out $dir/data_5000.model $dir/2000_best_path > $dir/2000_score
./svm_hmm_classify -v 0 --n 2 $dir/test.out $dir/data_5000.model $dir/2_best_path
