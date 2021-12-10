//mex CFLAGS='$CFLAGS -Ofast -march=native -ffast-math -Wall -funroll-loops -Wno-unused-result' sgupdate_node_embs_fast.c

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
double *vals_old;
double *vals_new;
int *flag_ini;
int *flag_current;
int *flag_next;

long long K_hash;
long long order_max;
double alpha_Katz;
double weight;


// void cws_fast_o1(int* affected_nodes, int num_affected_nodes){
// //    efficient implementation with sparse input for efficiency, 0.66s for blogcatalog,
// //     while 12s for building a full vector and sketch
//     double v_temp, v_min;
//     long long ind_n, ind_k, ind_emb, flag_sketch_change;
// //     long long counter=0,counter_bak;
//
//     for(int i=0; i<num_n; i++){
//         printf("flag %d\n",flag_ini[i]);
//         if (flag_ini[i]!=1){
// //             counter = counter+jc[i+1]-jc[i];
//         }
//         else{
//             ind_n = i*num_n;
//             ind_emb = i*K_hash;
//             flag_sketch_change=0;
// //             printf("sketching node %d\n",i);
//
// //             counter_bak = counter;
//             for(int k=0;k<K_hash;k++){
//                 v_min=vals_new[ind_emb+k];
//                 ind_k = num_m*k;
//
//                 for(int j=0; j<num_affected_nodes; j++){
//                     v_temp = rand_beta[ind_k+affected_nodes[j]]/1; // 1 means all incoming edges has weight 1
//                     if(v_temp<v_min){
//                         v_min = v_temp;
//                         embs_new[ind_emb+k] = affected_nodes[j]+1;
//                         flag_sketch_change=1;
//                     }
// //                     counter++;
//                 }
//
// //                 if (k<K_hash-1) {
// // //                 printf("node seen: %d\n",counter-counter_bak);
// //                     counter = counter_bak;
// //
// //                 }
//                 vals_new[ind_emb+k] = v_min;
// //             printf("K_hash %d\n",k);
// //             printf("v_min, ind_m: %f, %f\n",v_min,ind_m);
//             }
//             if (flag_sketch_change==1)
//                 for(int j=jc[i]; j<jc[i+1]; j++){
//                     flag_next[ir[j]]=1;
// //                     printf("neighbors %d\n",ir[j]);
//                 }
//         }
//     }
// }

void cws_fast_o1(){
    double v_temp, v_min;
    long long ind_n, ind_k, ind_emb, flag_sketch_change;
    long long counter=0,counter_bak;

    for(int i=0; i<num_n; i++){
//         printf("flag %d\n",flag_ini[i]);
        if (flag_ini[i]!=1){
            counter = counter+jc[i+1]-jc[i];
        }
        else{
            ind_n = i*num_n;
            ind_emb = i*K_hash;
            flag_sketch_change=0;
//             printf("sketching node %d\n",i);
            
            counter_bak = counter;
            for(int k=0;k<K_hash;k++){
                v_min=vals_new[ind_emb+k];
                ind_k = num_m*k;
                
                for(int j=jc[i]; j<jc[i+1]; j++){
                    v_temp = rand_beta[ind_k+ir[counter]]/network[counter];
                    if(v_temp<v_min){
                        v_min = v_temp;
                        embs_new[ind_emb+k] = ir[counter]+1;
                        flag_sketch_change=1;
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
            if (flag_sketch_change==1)
                for(int j=jc[i]; j<jc[i+1]; j++){
                    flag_next[ir[j]]=1;
//                     printf("neighbors %d\n",ir[j]);
                }
        }
    }
}


void cws_fast_recursive(){
    double *vec = (double *)mxCalloc(num_m, sizeof(double)); //a node embedding
    double v_temp, v_min;
    long long ind_n, ind_k, ind_emb, ind_s;
    long long counter=0,flag_sketch_change;
//     double counter_h=0;
    
    for(int i=0; i<num_n; i++){
        
        if (flag_current[i]!=1){
            counter = counter+jc[i+1]-jc[i];
        }
        else{
            ind_n = i*num_n;
            ind_emb = i*K_hash;
// build vec
            flag_sketch_change=0;
//             printf("sketching node %d\n",i);
            memset(vec, 0, num_m*sizeof(double)); //fast reset vec to zeros
            for(int j=jc[i]; j<jc[i+1]; j++) {
                vec[ir[counter]] += network[counter];
                ind_s = ir[counter]*K_hash;
                for(int s=0; s<K_hash; s++){
                    vec[(long long) (embs_old[ind_s+s]-1)] += weight;
//                     printf("ind_s+s/embs_old[ind_s+s] %d/%f\n",ind_s+s, embs_old[ind_s+s]-1);
                }
                counter++;
            }
            
            for(int k=0;k<K_hash;k++){
                v_min=INFINITY;
                ind_k = num_m*k;
                
                for(int j=0;j<num_m;j++){
//                     printf("vec[j] is %f\n",vec[j]);
                    if(vec[j]!=0){
//                         counter_h++;
                        v_temp = rand_beta[ind_k+j]/vec[j];
                        if(v_temp<v_min){
                            v_min = v_temp;
                            embs_new[ind_emb+k] = j+1;
                            flag_sketch_change=1;
                        }
//                         if((i==2)&&(k==2))
//                             printf("rand_beta[ind_k+j]/vec[j]/values/ are  %f/%f/%f\n",rand_beta[ind_k+j],v_temp,vec[j]);
                    }
                }
                vals_new[ind_emb+k] = v_min;
            }
//             printf("flag_sketch_change %d\n",flag_sketch_change);
            if (flag_sketch_change==1)
                for(int j=jc[i]; j<jc[i+1]; j++){
                    flag_next[ir[j]]=1;
//                     printf("neighbors %d\n",ir[j]);
                }
        }
    }
//     printf("counter_ox: %f\n",counter_h/K_hash);
}


void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 9) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "9 inputs required.");
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
    
    
    
    double * edges = (double *)mxGetData(prhs[7]);
    long long num_edge_m = mxGetM(prhs[7]);
    long long num_edge_n = mxGetN(prhs[7]);
    
    double verbose = mxGetScalar(prhs[8]);
//     print arguments
    if (verbose==1){
        mexPrintf("num of nodes (rows): %lld; num of nodes (columns): %lld; embedding dimension: %lld\n", num_m,num_n, K_hash);
        mexPrintf("rand_m (rows) : %lld; rand_n: %lld\n",  rand_m, rand_n);
        mexPrintf("order_max: %lld\n",order_max);
        mexPrintf("alpha_Katz: %f\n",alpha_Katz);
        mexPrintf("nodes in a edges: %lld\n",num_edge_m);
        mexPrintf("num of edges: %lld\n",num_edge_n);
        mexPrintf("num of edges: %lld\n",num_edge_m);
        if (order_max<1){
            mexErrMsgIdAndTxt("SGSketch:order",
                    "order should be an integer >=1");
        }
        fflush(stdout);
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
    
    flag_ini = (int *)mxCalloc(num_n, sizeof(int)); // initialized with 0
    flag_current = (int *)mxCalloc(num_n, sizeof(int));
    flag_next = (int *)mxCalloc(num_n, sizeof(int));
//     int *affected_nodes = (int *)mxCalloc(num_n, sizeof(int));
    int edge_ind = 0; //num_affected_nodes=0;
    for(int j=0; j<num_edge_n*num_edge_m; j++) {
        edge_ind = edges[j]-1;
        flag_ini[edge_ind] = 1; // MATLAB edge index to C index --
    }
//     for(int i=0; i<num_n; i++)
//         if (flag_ini[i]==1)
//             affected_nodes[num_affected_nodes++] = i;
    
    
    memcpy(flag_next, flag_ini, num_n*sizeof(int));
    embs_old = (double *)mxMalloc(K_hash*num_n*sizeof(double));
    
//     memset(flag, 0, num_n*sizeof(double));
    
    if (verbose==1)
        printf("sketching starts ...\n");
    
    for (int i=0;i<order_max;i++){
//         embs_temp = mxGetPr(mxGetCell(prhs[5],i));
//         vals_temp = mxGetPr(mxGetCell(prhs[6],i));
        memcpy(embs_new, mxGetPr(mxGetCell(prhs[5],i)), K_hash*num_n*sizeof(double));
        memcpy(vals_new, mxGetPr(mxGetCell(prhs[6],i)), K_hash*num_n*sizeof(double));
        if (i==0){
            cws_fast_o1();
            memcpy(flag_current, flag_next, num_n*sizeof(int));
            memcpy(embs_old, embs_new, K_hash*num_n*sizeof(double));
        }
        else{
            if (verbose==1)
                printf("sketching starts (order %d...)\n",i+1);
            
            memcpy(flag_next, flag_ini, num_n*sizeof(int));
            cws_fast_recursive();
            memcpy(flag_current, flag_next, num_n*sizeof(int));
            memcpy(embs_old, embs_new, K_hash*num_n*sizeof(double));
        }
//         for(int j=0; j<num_n*K_hash; j++) embs_new[j]++;
        mxSetCell(plhs[0],i,mxDuplicateArray(embs_new_mat));
        mxSetCell(plhs[1],i,mxDuplicateArray(vals_new_mat));
//         for(int j=0; j<num_n*K_hash; j++) embs_new[j]--;
        
    }
    
    
    
}






