echo "Retrofit with 50000 test phrases"
python constituentretrofit_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin -t 50000
echo "Evaluate It"
python evaluate_rank_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin.cons -t data/testVocab.json

echo "Retrofit with 10000 test phrases"
python constituentretrofit_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin -t 10000
echo "Evaluate It"
python evaluate_rank_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin.cons -t data/testVocab.json

echo "Retrofit with 10000 test phrases"
python constituentretrofit_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin -t 10000
echo "Evaluate It"
python evaluate_rank_fixed_word2vec.py -v data/GoogleNews-vectors-negative300.bin.cons -t data/testVocab.json
