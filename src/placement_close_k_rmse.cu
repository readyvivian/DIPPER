#include "mash_placement.cuh"
#include "placement_close_k.cuh"

#include <stdio.h>
#include <queue>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cub/cub.cuh>
#include <limits.h>

struct compare_tuple_RMSE {
  __host__ __device__
  bool operator()(thrust::tuple<int,double,double,double> lhs, thrust::tuple<int,double,double,double> rhs)
  {
    return thrust::get<3>(lhs) < thrust::get<3>(rhs);
    //Always find the tuple whose third value (the criteria we want to minimize) is minimized
  }
};
/*
Four variables in tuple:
ID of branch in linked list,
distance to new node inserted on branch from starting vertex (belong[id]),
distance from new node inserted on branch to new node inserted outside branch
mse value for inserting the new node on the branch
*/

__global__ void calculateBranchLengthRMSE(
    int num, // should be bd, not numSequences 
    int * head,
    int * nxt,
    double * dis, 
    int * e, 
    double * len, 
    int * belong,
    thrust::tuple<int,double,double, double> * minPos,
    int lim,
    double * closest_dis,
    int * closest_id,
    int totSeqNum
){
    int tx=threadIdx.x,bs=blockDim.x,bx=blockIdx.x,gs=gridDim.x;
    int idx_=tx+bs*bx;
    for (int idx=idx_; idx<lim; idx+=bs*gs){
        if(idx>=lim) return;
        if(idx>=num*4-4||belong[idx]<e[idx]){
            thrust::tuple <int,double,double,double> minTuple(0,0,2,INT_MAX);
            minPos[idx]=minTuple;
            continue;
        }
        int x=belong[idx],oth=e[idx];
        int eid=idx,otheid;
        double mean_u=0, mean_v=0, val;
        int cnt_u=0, cnt_v=0;
        for(int i=0;i<5;i++)
            if(closest_id[eid*5+i]!=-1){
                val = dis[closest_id[eid*5+i]]-closest_dis[eid*5+i];
                mean_u += val;
                cnt_u++;
            }
        if(cnt_u>0) mean_u /= cnt_u;
        otheid=head[oth];
        while(e[otheid]!=x) assert(otheid!=-1),otheid=nxt[otheid];
        for(int i=0;i<5;i++)
            if(closest_id[otheid*5+i]!=-1){
                val = dis[closest_id[otheid*5+i]]-closest_dis[otheid*5+i];
                mean_v += val;
                cnt_v++;
            }
        if(cnt_v>0) mean_v /= cnt_v;

        double additional_dis=(mean_u+mean_v-len[eid])/2;
        double dis1=(len[eid]+mean_u-mean_v)/2;
        double dis2=len[eid]-dis1;
        // clamp:
        if(additional_dis<0) additional_dis=0;
        if(dis1<0) dis1=0;
        if(dis2<0) dis2=0;
        if(dis1>len[eid]) additional_dis+=dis1-len[eid],dis1=len[eid];
        if(dis2>len[eid]) additional_dis+=dis2-len[eid],dis2=len[eid];
        double rest=len[eid]-dis1-dis2;
        dis1+=rest/2,dis2+=rest/2;
        additional_dis = ((mean_u-dis1)+(mean_v-dis2))/2;
        if(additional_dis<0) additional_dis=0;
        
        double mse = 0;
        int cnt = 0;
        for(int i=0;i<5;i++)
            if(closest_id[eid*5+i]!=-1){
                val = additional_dis+dis1-dis[closest_id[eid*5+i]]+closest_dis[eid*5+i];
                mse += val*val;
                cnt++;
            }
        for(int i=0;i<5;i++)
            if(closest_id[otheid*5+i]!=-1){
                val = additional_dis+dis2-dis[closest_id[otheid*5+i]]+closest_dis[otheid*5+i];
                mse += val*val;
                cnt++;
            }
        if (cnt>0) mse /= cnt;
        thrust::tuple <int,double,double,double> minTuple(eid,dis1,additional_dis,mse);
        minPos[idx]=minTuple;
    }
}

void MashPlacement::KPlacementDeviceArrays::findPlacementTreeRMSE(
    Param& params,
    const MashDeviceArrays& mashDeviceArrays,
    MatrixReader& matrixReader,
    const MSADeviceArrays& msaDeviceArrays
){ 
    if(params.in == "d"){
        matrixReader.distConstructionOnGpu(params, 0, d_dist);
    }
    int * d_id;
    auto err = cudaMalloc(&d_id, numSequences*2*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    int * d_from;
    err = cudaMalloc(&d_from, numSequences*2*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    double * d_dis;
    err = cudaMalloc(&d_dis, numSequences*2*sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    
    /*
    Initialize closest nodes by inifinite
    */
    // int threadNum = 256, blockNum = (numSequences*4-4+threadNum-1)/threadNum;
    int threadNum = 1024, blockNum = 1024;
    initialize <<<blockNum, threadNum>>> (
        numSequences*4-4,
        numSequences*2,
        d_closest_dis,
        d_closest_id,
        d_head,
        d_nxt,
        d_belong,
        d_e
    );
    /*
    Build Initial Tree
    */
    if(params.in == "r"){
        mashDeviceArrays.distConstructionOnGpu(
            params,
            1,
            d_dist
        );
    }
    else if(params.in == "d"){
        matrixReader.distConstructionOnGpu(
            params,
            1,
            d_dist
        );
    }
    else if(params.in == "m"){
        msaDeviceArrays.distConstructionOnGpu(
            params,
            1,
            d_dist
        );
    }
    // cudaDeviceSynchronize();

    // return;
    // double * h_dis = new double[numSequences];
    // cudaMemcpy(h_dis,d_dist,numSequences*sizeof(double),cudaMemcpyDeviceToHost);
    // for(int j=0;j<1;j++) fprintf(stderr,"%.8lf ",h_dis[j]);std::cerr<<'\n';

    buildInitialTree <<<1,1>>> (
        numSequences,
        d_head,
        d_e,
        d_len,
        d_nxt,
        d_belong,
        d_dist,
        idx
    );
    idx += 4;
    /*
    Initialize closest nodes by inital tree
    */
    for(int i=0;i<bd;i++){
        updateClosestNodes <<<1,1>>> (
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            i,
            d_id,
            d_from,
            d_dis
        );
    }
 
    thrust::device_vector <thrust::tuple<int,double,double,double>> minPos(numSequences*4-4);
    // std::cout<<"FFF\n";
    std::chrono::nanoseconds disTime(0), treeTime(0);
    for(int i=bd;i<numSequences;i++){
        auto disStart = std::chrono::high_resolution_clock::now();
        // blockNum = (i + 255) / 256;
        // blockNum = 1024;
        if(params.in == "r"){
            mashDeviceArrays.distConstructionOnGpu(
                params,
                i,
                d_dist
            );
        }
        else if(params.in == "d"){
            matrixReader.distConstructionOnGpu(
                params,
                i,
                d_dist
            );
        }
        else if(params.in == "m"){
            msaDeviceArrays.distConstructionOnGpu(
                params,
                i,
                d_dist
            );
        }
        cudaDeviceSynchronize();

        // double * h_dis = new double[numSequences];
        // cudaMemcpy(h_dis,d_dist,numSequences*sizeof(double),cudaMemcpyDeviceToHost);
        // fprintf(stderr, "%d\n",i);
        // for(int j=0;j<i;j++) std::cerr<<h_dis[j]<<" ";std::cerr<<'\n';


        auto disEnd = std::chrono::high_resolution_clock::now();
        auto treeStart = std::chrono::high_resolution_clock::now();
        
        calculateBranchLengthRMSE <<<blockNum,threadNum>>> (
            i,
            d_head,
            d_nxt,
            d_dist,
            d_e,
            d_len,
            d_belong,
            thrust::raw_pointer_cast(minPos.data()),
            numSequences*4-4,
            d_closest_dis,
            d_closest_id,
            numSequences
        );
        
        auto iter=thrust::min_element(minPos.begin(),minPos.end(),compare_tuple_RMSE());
        thrust::tuple<int,double,double,double> smallest=*iter;
        // /* print top 5 sorted elements */
        // for(int j=0;j<5;j++){
        //     thrust::tuple<int,double,double> s=*iter;
        //     std::cerr<<thrust::get<0>(s)<<" "<<thrust::get<1>(s)<<" "<<thrust::get<2>(s)<<'\n';
        //     iter++;
        // } 
        thrust::tuple<int,double,double,double> s = smallest;
        std::cerr<<thrust::get<0>(s)<<" "<<thrust::get<1>(s)<<" "<<thrust::get<2>(s)<<" "<<thrust::get<3>(s)<<'\n';
        
        /*
        Update Tree (and assign closest nodes to newly added nodes)
        */
        int eid=thrust::get<0>(smallest);
        double fracLen=thrust::get<1>(smallest),addLen=thrust::get<2>(smallest);
        updateTreeStructure <<<1,1>>>(
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            d_belong,
            eid,
            fracLen,
            addLen,
            i,
            idx,
            numSequences
        );
        idx+=4;
        /*
        Update closest nodes
        */
        updateClosestNodes <<<1,1>>> (
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            i,
            d_id,
            d_from,
            d_dis
        );
        // cudaDeviceSynchronize();
        auto treeEnd = std::chrono::high_resolution_clock::now();
        disTime += disEnd - disStart;
        treeTime += treeEnd - treeStart;
        // std::cerr << "Seq " << i << " at " << eid << " with fracLen " << fracLen 
        //           << " and addLen " << addLen << std::endl;
    }
    std::cerr << "Distance Operation Time " <<  disTime.count()/1000000 << " ms\n";
    std::cerr << "Tree Operation Time " <<  treeTime.count()/1000000 << " ms\n";
}