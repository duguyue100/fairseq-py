# This is a Python template Makefile, do modification as you want
#
# Project: 
# Author:
# Email :

HOST = 127.0.0.1
PYTHONPATH="$(shell printenv PYTHONPATH):$(PWD)"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

# EXPERIMENTS FOR EN2CN SUBWORD LEVEL
preproc-en2cn:
	python preprocess.py --source-lang en --target-lang cn --trainpref data/data_3mil/3mil.tok.train --validpref data/data_3mil/3mil.tok.valid --testpref data/data_3mil/3mil.tok.test --destdir data-bin/en2cn

## fconv experiments
run-1-en2cn-fconv:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_en2cn --save-dir checkpoints/fconv-en2cn-run-1 --max-epoch 20

run-2-en2cn-fconv:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_en2cn --save-dir checkpoints/fconv-en2cn-run-2 --max-epoch 20

run-3-en2cn-fconv:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_en2cn --save-dir checkpoints/fconv-en2cn-run-3 --max-epoch 20

run-4-en2cn-fconv:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_en2cn --save-dir checkpoints/fconv-en2cn-run-4 --max-epoch 20

run-5-en2cn-fconv:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch fconv_en2cn --save-dir checkpoints/fconv-en2cn-run-5 --max-epoch 20

## GRU experiments
run-1-en2cn-gru:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch GRU_en2cn_model --save-dir checkpoints/gru-en2cn-run-1 --max-epoch 20

run-2-en2cn-gru:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch GRU_en2cn_model --save-dir checkpoints/gru-en2cn-run-2 --max-epoch 20

run-3-en2cn-gru:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch GRU_en2cn_model --save-dir checkpoints/gru-en2cn-run-3 --max-epoch 20

run-4-en2cn-gru:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch GRU_en2cn_model --save-dir checkpoints/gru-en2cn-run-4 --max-epoch 20

run-5-en2cn-gru:
	python train.py data-bin/en2cn --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 1000 --arch GRU_en2cn_model --save-dir checkpoints/gru-en2cn-run-5 --max-epoch 20

# EXPERIMENTS ON CN2EN
preproc-cn2en:
	python preprocess.py --source-lang cn --target-lang en --trainpref data/data_3mil/3mil.tok.train --validpref data/data_3mil/3mil.tok.valid --testpref data/data_3mil/3mil.tok.test --destdir data-bin/cn2en

preproc-en2wb:
	python preprocess.py --source-lang en --target-lang wb --trainpref data/data_3mil/3mil.tok.train --validpref data/data_3mil/3mil.tok.valid --testpref data/data_3mil/3mil.tok.test --destdir data-bin/en2wb

preproc-wb2en:
	python preprocess.py --source-lang wb --target-lang en --trainpref data/data_3mil/3mil.tok.train --validpref data/data_3mil/3mil.tok.valid --testpref data/data_3mil/3mil.tok.test --destdir data-bin/wb2en

preproc-en2wubi-char:

preproc-wubi2en-char:

test:
	PYTHONPATH=$(PYTHONPATH) python 

cleanall:
