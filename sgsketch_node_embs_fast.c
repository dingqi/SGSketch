//mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sgsketch_node_embs_fast.c

#include "mex.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"
#include "limits.h"
#include "string.h"

double *rand_beta;
mwSize rand_n, rand_m;

double *network;
mwSize num_n, num_m;
mwIndex  *ir, *jc;

double *embs_old;
double *embs_new;
double *vals_new;

long long K_hash;
long long order_max;
double alpha_Katz;
double weight;


void cws_fast_o1(){
//    efficient implementation with sparse input for efficiency, 0.66s for blogcatalog, while 12s for building a full vector and sketch
    double v_temp, v_min;
    long long ind_n, ind_k, ind_emb;
    long long counter=0,counter_bak;
    
    for(int i=0; i<num_n; i++){
        ind_n = i*num_n;
        ind_emb = i*K_hash;
//         printf("sketching node %d\n",i);
        
        counter_bak = counter;
        for(int k=0;k<K_hash;k++){
            v_min=INFINITY;
            ind_k = num_m*k;
            
            for(int j=jc[i]; j<jc[i+1]; j++){
                v_temp = rand_beta[ind_k+ir[counter]]/network[counter];
                if(v_temp<v_min){
                    v_min = v_temp;
                    embs_new[ind_emb+k] = ir[counter];
                }
                counter++;
            }
            
            if (k<K_hash-1) {
//                 printf("node seen: %d\n",counter-counter_bak);
                counter = counter_bak;
                
            }
            vals_new[ind_emb+k] = v_min;
//             printf("K_hash %d\n",k);
//             printf("v_min, ind_m: %f, %f\n",v_min,ind_m);
        }
    }
//     printf("counter_o1: %d\n", counter);
//     printf("cws_fast_o1 done!\n");
}


void cws_fast_recursive(){
    double *vec = (double *)mxCalloc(num_m, sizeof(double)); //a node embedding
    double v_temp, v_min;
    long long ind_n, ind_k, ind_emb, ind_s;
    long long counter=0;
    double counter_h=0;
    
    for(int i=0; i<num_n; i++){
        ind_n = i*num_n;
        ind_emb = i*K_hash;
        
// build vec
        memset(vec, 0, num_m*sizeof(double)); //fast reset vec to zeros
        for(int j=jc[i]; j<jc[i+1]; j++) {
            vec[ir[counter]] += network[counter];
            ind_s = ir[counter]*K_hash;
            for(int s=0; s<K_hash; s++){
                vec[(long long) (embs_old[ind_s+s])] += weight;
            }
            counter++;
        }
        
        
        for(int k=0;k<K_hash;k++){
            v_min=INFINITY;
            ind_k = num_m*k;
            
            for(int j=0;j<num_m;j++){
//                 printf("vec[j] is %f\n",vec[j]);
                
                if(vec[j]!=0){
                    counter_h++;
                    v_temp = rand_beta[ind_k+j]/vec[j];
                    if(v_temp<v_min){
                        v_min = v_temp;
                        embs_new[ind_emb+k] = j;
                    }
//                     if((i==2)&&(k==2))
//                             printf("rand_beta[ind_k+j]/vec[j]/values/ are  %f/%f/%f\n",rand_beta[ind_k+j],v_temp,vec[j]);
                }
            }
            vals_new[ind_emb+k] = v_min;
        }
    }
//     printf("counter_ox: %f\n",counter_h/K_hash);
}


void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "6 inputs required.");
        //(network, K_hash, Rand_beta, order_max, alpha)
    }
    if(nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                "2 output required.");
    }
    
    network = (double *)mxGetData(prhs[0]); // read from file
    num_m = mxGetM(prhs[0]);
    num_n = mxGetN(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    
    
    
    K_hash = mxGetScalar(prhs[1]);
    
    rand_beta = (double *)mxGetData(prhs[2]);
    rand_m = mxGetM(prhs[2]);
    rand_n = mxGetN(prhs[2]);
    
    
    order_max = mxGetScalar(prhs[3]);
    alpha_Katz = mxGetScalar(prhs[4]);
    
    weight = alpha_Katz/K_hash;
    
    double verbose = mxGetScalar(prhs[5]);
    
//     print arguments
    if (verbose==1){
        mexPrintf("num of nodes (rows): %lld; num of nodes (columns): %lld; embedding dimension: %lld\n", num_m,num_n, K_hash);
        mexPrintf("rand_m (rows) : %lld; rand_n: %lld\n",  rand_m, rand_n);
        mexPrintf("order_max: %lld\n",order_max);
        mexPrintf("alpha_Katz: %f\n",alpha_Katz);
        fflush(stdout);
        
        if (order_max<1){
            mexErrMsgIdAndTxt("SGSketch:order",
                    "order should be an integer >=1");
        }
    }
    
    
//     plhs[0] = mxCreateDoubleMatrix(K_hash,num_n,mxREAL);
//     plhs[1] = mxCreateDoubleMatrix(K_hash,num_n,mxREAL);
//
//     embs_new = mxGetPr(plhs[0]);
//     vals_new = mxGetPr(plhs[1]);
    
    plhs[0] = mxCreateCellMatrix(order_max,1);
    plhs[1] = mxCreateCellMatrix(order_max,1);
    
    mxArray * embs_new_mat = mxCreateDoubleMatrix(K_hash,num_n,mxREAL);
    mxArray * vals_new_mat = mxCreateDoubleMatrix(K_hash,num_n,mxREAL);
    embs_new = mxGetPr(embs_new_mat);
    vals_new = mxGetPr(vals_new_mat);
    
    if (verbose==1)
        printf("sketching starts ...\n");
    
    for (int i=0;i<order_max;i++){
        if (i==0){
            cws_fast_o1();
        }
        else{
            embs_old = (double *)mxMalloc(K_hash*num_n*sizeof(double));
            memcpy(embs_old, embs_new, K_hash*num_n*sizeof(double));
            if (verbose==1)
                printf("sketching starts (order %d...)\n",i+1);
            cws_fast_recursive();
        }
        
        
        for(int j=0; j<num_n*K_hash; j++) embs_new[j]++;
        mxSetCell(plhs[0],i,mxDuplicateArray(embs_new_mat));
        mxSetCell(plhs[1],i,mxDuplicateArray(vals_new_mat));
        for(int j=0; j<num_n*K_hash; j++) embs_new[j]--;
        
    }
    
    
    
}






