# GCIL
The official implementation of the TIST paper: Empowering Predictive Modeling by GAN-based Causal Information Learning.

An example config command:
```
python main.py --dataset_path Beibei_normed_ood1.npy -c 1 --model 'main' --causal_test 0 --disentangle 1 -beta 0.09 -lr 0.005 -lra 0.001 --batch_size 128 --base Densenet


