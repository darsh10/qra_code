Repository for Adversarial Domain Adaptation for Duplicate Question Detection (https://www.aclweb.org/anthology/D18-1131/) EMNLP 2018.

Requirements: Pytorch 0.1.11

For questions, email darsh@csail.mit.edu


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

--------------------------------------------------------------------------------

If you find this repository helpful, please cite our paper:
```
@inproceedings{shah-etal-2018-adversarial,
    title = "Adversarial Domain Adaptation for Duplicate Question Detection",
    author = "Shah, Darsh  and
      Lei, Tao  and
      Moschitti, Alessandro  and
      Romeo, Salvatore  and
      Nakov, Preslav",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1131",
    doi = "10.18653/v1/D18-1131",
    pages = "1056--1063",
    abstract = "We address the problem of detecting duplicate questions in forums, which is an important step towards automating the process of answering new questions. As finding and annotating such potential duplicates manually is very tedious and costly, automatic methods based on machine learning are a viable alternative. However, many forums do not have annotated data, i.e., questions labeled by experts as duplicates, and thus a promising solution is to use domain adaptation from another forum that has such annotations. Here we focus on adversarial domain adaptation, deriving important findings about when it performs well and what properties of the domains are important in this regard. Our experiments with StackExchange data show an average improvement of 5.6{\%} over the best baseline across multiple pairs of domains.",
}
