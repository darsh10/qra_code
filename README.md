Repository for question similarity with domain adaptation.

Requirements: Pytorch 0.1.11

MIT License for all resources.

Data is available at https://drive.google.com/file/d/1H26LUWeso0gT8TUvLd7W23WGBjyoAS7z/view?usp=sharing

Word vectors used for the project are available at https://drive.google.com/file/d/177x3qz0IFHGSHDzULNESfu_vzXK9i944/view?usp=sharing

To run the direct model, a sample command is python main.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine

To run the adversarial model, a sample command is python main_domain.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine --cross_domain ../android/ --wasserstein
