# tf-eager-nlm
Neural Language Modeling using tf.eager

Adapted from [the official tutorial on using eager to build Language Model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/rnn_ptb/rnn_ptb.py)

A verbose and slow paced [blog post is available on Medium](https://medium.com/@vineet.mundhra/building-a-neural-language-model-ee3090e4e312)

#### Installation
Clone the repo, create a new environment and install requirements.txt
```bash
  git clone git@github.com:vineetm/tf-eager-nlm.git
  cd tf-eager-nlm

  conda create -n tfnlm python=3.6
  source activate tfnlm
  (tfnlm) pip install -r requirements.txt
```