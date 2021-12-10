# SGSketch
This is the implementation for SGSketch in MATLAB with C (see the following paper):

Dingqi Yang, Bingqing Qu, Jie Yang, Liang Wang, and Philippe Cudre-Mauroux, Streaming Graph Embeddings via Incremental Neighborhood Sketching, 2021 (under review)

How to use (Tested on MATLAB 2017b on MacOS and Ubuntu):
1. Compile sgsketch_node_embs_fast.c and sgupdate_node_embs_fast.c using mex (in MATLAB): 
  - mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sgsketch_node_embs_fast.c
  - mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sgupdate_node_embs_fast.c

2. Run experiment.m

Please cite our paper if you publish material using this code. 
