Repository for question similarity with domain adaptation.

Requirements: Pytorch 0.1.11

For questions, email darsh@csail.mit.edu

-----------------------------------------------------------------------------
The MIT License

------------------------------------------------------------------------------

To run the direct model, a sample command is 

python main.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/all_corpora_vectors.txt --max_epoch 50 --use_content --eval_use_content --criterion cosine

-------------------------------------------------------------------------------

To run the adversarial model, a sample command is 

python main_domain.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/all_corpora_vectors.txt --max_epoch 50 --use_content --eval_use_content --criterion cosine --cross_train ../android/ --wasserstein

-------------------------------------------------------------------------------

Data is available at https://drive.google.com/file/d/1blckd0P8KX0FHCA8tBlkJ5hmYhrO5DcX/view?usp=sharing

-------------------------------------------------------------------------------
Embeddings: https://drive.google.com/file/d/17RcM_senbkcurXOUg-864LZAsn3OiTjM/view?usp=sharing

-------------------------------------------------------------------------------
The hyper-parameters of the baseline model are the following:  number of hidden dimensions of the LSTM 100, the dropout {0.0, 0.1}, learning rate {0.0001, 0.0005, 0.001}, and the number of layers of LSTMs 1.
