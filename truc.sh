CUDA_VISIBLE_DEVICES=0 python3 decisionTransformer.py 8 0 4 8 &
CUDA_VISIBLE_DEVICES=1 python3 decisionTransformer.py 8 1 5 9 &
CUDA_VISIBLE_DEVICES=2 python3 decisionTransformer.py 8 2 6 &
CUDA_VISIBLE_DEVICES=3 python3 decisionTransformer.py 8 3 7 &