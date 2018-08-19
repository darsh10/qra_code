Repository for question similarity with domain adaptation.

Requirements: Pytorch 0.1.11

For questions, email darsh@csail.mit.edu

The MIT License

Copyright (c) 2010-2018 Google, Inc. http://angularjs.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Data is available at https://drive.google.com/file/d/1H26LUWeso0gT8TUvLd7W23WGBjyoAS7z/view?usp=sharing

Word vectors used for the project are available at https://drive.google.com/file/d/177x3qz0IFHGSHDzULNESfu_vzXK9i944/view?usp=sharing

To run the direct model, a sample command is 

python main.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine


To run the adversarial model, a sample command is 

python main_domain.py --cuda --run_dir /tmp/ --model lstm --train ../askubuntu/train --eval ../android/ --bidir --d 100 --embedding ../word_vectors/glove.npz --max_epoch 50 --use_content --eval_use_content --criterion cosine --cross_domain ../android/ --wasserstein
