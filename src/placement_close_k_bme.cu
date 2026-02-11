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

__global__ void updateClosestNodesBME(
    int * head,
    int * nxt,
    int * e,
    double * len,
    double * closest_dis,
    int * closest_id,
    int * closest_branches,
    int x,
    int * id,
    int * from,
    double * dis
){
    int l=0,r=-1;
    // id[++r]=x,dis[x]=0,from[x]=-1;
    id[++r]=x,dis[r]=0,from[r]=-1;
    int level=1, curr_end=r, next_end=r;
    while(l<=r){
        if (l>curr_end) {level++; curr_end=next_end;}
        int node=id[l],fb=from[l];
        double d=dis[l];
        l++;
        for(int i=head[node];i!=-1;i=nxt[i]){
            // printf("%d %d: \n", node, head[node]);
            if(e[i]==fb) continue;
            for(int j=0;j<5;j++){
                double nowd=closest_dis[i*5+j];
                if(nowd>d){
                    for(int k=4;k>j;k--){
                        closest_dis[i*5+k]=closest_dis[i*5+k-1];
                        closest_id[i*5+k]=closest_id[i*5+k-1];
                        closest_branches[i*5+k]=closest_branches[i*5+k-1];
                    }
                    // printf("%d: (%d %lf)\t", i*5+j, x, d);
                    printf("%d: level %d\n", i*5+j, level);
                    closest_dis[i*5+j]=d;
                    closest_id[i*5+j]=x;
                    closest_branches[i*5+j]=level;
                    id[++r]=e[i],dis[r]=d+len[i],from[r]=node,next_end=r;
                    break;
                }
            }
            // printf("\n");
        }
    }
}

__global__ void initializeBME(
    int lim,
    int nodes,
    double * d_closest_dis,
    int * d_closest_id,
    int * d_closest_branches,
    int * head,
    int * nxt,
    int * belong,
    int * e
){
    int tx=threadIdx.x,bs=blockDim.x;
    int bx=blockIdx.x,gs=gridDim.x;
    int idx=tx+bs*bx;
    for (int t=idx; t<lim; t+=bs*gs){
        if(t<lim){
            for(int i=0;i<5;i++){
                d_closest_dis[t*5+i]=2;
                d_closest_id[t*5+i]=-1;
                d_closest_branches[t*5+i]=INT_MAX;
            }
            nxt[t] = -1;
            e[t] = -1;
            belong[t] = -1;
        }
        if(t<nodes) head[t] = -1;
    }
}

struct compare_tuple_BME {
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
bme score for inserting the new node on the branch
*/

__global__ void calculateBranchLengthBME(
    int num, // should be bd, not numSequences 
    int * head,
    int * nxt,
    double * dis, 
    int * e, 
    double * len, 
    int * belong,
    thrust::tuple<int,double,double,double> * minPos,
    int lim,
    double * closest_dis,
    int * closest_id,
    int * closest_branches,
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
        double dis1=0, dis2=0, val, bme_score=0;
        int valid_closest=0;
        for(int i=0;i<5;i++)
            if(closest_id[eid*5+i]!=-1){
                val = dis[closest_id[eid*5+i]]-closest_dis[eid*5+i];
                if(val>dis1) dis1=val;
                valid_closest++;
                bme_score += dis[closest_id[eid*5+i]] * pow(0.5, closest_branches[eid*5+i]);
            }
        otheid=head[oth];
        while(e[otheid]!=x) assert(otheid!=-1),otheid=nxt[otheid];
        for(int i=0;i<5;i++)
            if(closest_id[otheid*5+i]!=-1){
                val = dis[closest_id[otheid*5+i]]-closest_dis[otheid*5+i];
                if(val>dis2) dis2=val;
                valid_closest++;
                bme_score += dis[closest_id[otheid*5+i]] * pow(0.5, closest_branches[otheid*5+i]);
            }
        double additional_dis=(dis1+dis2-len[eid])/2;
        if(additional_dis<0) additional_dis=0;
        dis1-=additional_dis,dis2-=additional_dis;
        if(dis1<0) dis1=0;
        if(dis2<0) dis2=0;
        if(dis1>len[eid]) additional_dis+=dis1-len[eid],dis1=len[eid];
        if(dis2>len[eid]) additional_dis+=dis2-len[eid],dis2=len[eid];
        // assert(dis1+dis2-1e-6<=len[eid]);
        double rest=len[eid]-dis1-dis2;
        dis1+=rest/2,dis2+=rest/2;
        // print eif, dis1, and additional_dis
        // printf("eid %d (nodes %d-%d): dis1=%.8lf, dis2=%.8lf, len=%.8lf, add=%.8lf\n", eid, x, oth, dis1, dis2, len[eid], additional_dis);
        if (valid_closest == 0) {
            thrust::tuple <int,double,double,double> minTuple(0,0,2,INT_MAX);
            minPos[idx]=minTuple; continue;
        }
        // bme_score /= valid_closest;
        thrust::tuple <int,double,double,double> minTuple(eid,dis1,additional_dis,bme_score);
        minPos[idx]=minTuple;
    }
}

__global__ void updateTreeStructureBME(
    int * head,
    int * nxt,
    int * e,
    double * len,
    double * closest_dis,
    int * closest_id,
    int * closest_branches,
    int * belong,
    int eid,
    double fracLen,
    double addLen,
    int placeId, // Id of the newly placed node
    int edgeCount, // Position to insert a new edge in linked list
    int numSequences
){
    int middle=placeId+numSequences-1, outside=placeId;
    int x=belong[eid],y=e[eid];
    double originalDis=len[eid];
    int xe,ye;
    for(int i=head[x];i!=-1;i=nxt[i])
        if(e[i]==y){
            e[i]=middle,len[i]=fracLen,xe=i;
            break;
        }
    for(int i=head[y];i!=-1;i=nxt[i])
        if(e[i]==x){
            e[i]=middle,len[i]-=fracLen,ye=i;
            break;
        }
    /*
    Need to update:
    e, len, nxt, head, belong, closest_dis, closest_id
    */
    //middle -> x
    e[edgeCount]=x,len[edgeCount]=fracLen,nxt[edgeCount]=head[middle],head[middle]=edgeCount,belong[edgeCount]=middle;
    for(int i=0;i<5;i++)
        if(closest_id[ye*5+i]!=-1){
            closest_id[edgeCount*5+i]=closest_id[ye*5+i];
            closest_dis[edgeCount*5+i]=closest_dis[ye*5+i]+originalDis-fracLen;
            int br = closest_branches[ye*5+i];
            closest_branches[edgeCount*5+i] = (br==INT_MAX)?INT_MAX:br+1;
        }
    edgeCount++;
    //middle -> y
    e[edgeCount]=y,len[edgeCount]=originalDis-fracLen,nxt[edgeCount]=head[middle],head[middle]=edgeCount,belong[edgeCount]=middle;
    for(int i=0;i<5;i++)
        if(closest_id[xe*5+i]!=-1){
            closest_id[edgeCount*5+i]=closest_id[xe*5+i];
            closest_dis[edgeCount*5+i]=closest_dis[xe*5+i]+fracLen;
            int br = closest_branches[xe*5+i];
            closest_branches[edgeCount*5+i] = (br==INT_MAX)?INT_MAX:br+1;
        }
    edgeCount++;
    //outside -> middle
    e[edgeCount]=middle,len[edgeCount]=addLen,nxt[edgeCount]=head[outside],head[outside]=edgeCount,belong[edgeCount]=outside;
    edgeCount++;
    //middle -> outside
    e[edgeCount]=outside,len[edgeCount]=addLen,nxt[edgeCount]=head[middle],head[middle]=edgeCount,belong[edgeCount]=middle;
    int e1=edgeCount-2, e2=edgeCount-3;
    for(int i=0;i<5;i++){
        if(closest_id[e1*5+i]==-1) break;
        for(int j=0;j<5;j++)
            if(closest_dis[edgeCount*5+j]>closest_dis[e1*5+i]){
                for(int k=4;k>j;k--){
                    closest_dis[edgeCount*5+k]=closest_dis[edgeCount*5+k-1];
                    closest_id[edgeCount*5+k]=closest_id[edgeCount*5+k-1];
                    closest_branches[edgeCount*5+k]=closest_branches[edgeCount*5+k-1];
                }
                closest_dis[edgeCount*5+j]=closest_dis[e1*5+i];
                closest_id[edgeCount*5+j]=closest_id[e1*5+i];
                closest_branches[edgeCount*5+j]=closest_branches[e1*5+i];
                break;
            }
    }
    for(int i=0;i<5;i++){
        if(closest_id[e2*5+i]==-1) break;
        for(int j=0;j<5;j++)
            if(closest_dis[edgeCount*5+j]>closest_dis[e2*5+i]){
                for(int k=4;k>j;k--){
                    closest_dis[edgeCount*5+k]=closest_dis[edgeCount*5+k-1];
                    closest_id[edgeCount*5+k]=closest_id[edgeCount*5+k-1];
                    closest_branches[edgeCount*5+k]=closest_branches[edgeCount*5+k-1];
                }
                closest_dis[edgeCount*5+j]=closest_dis[e2*5+i];
                closest_id[edgeCount*5+j]=closest_id[e2*5+i];
                closest_branches[edgeCount*5+j]=closest_branches[e2*5+i];
                break;
            }
    }
    edgeCount++;
}

void MashPlacement::KPlacementDeviceArrays::findPlacementTreeBME(
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
    initializeBME <<<blockNum, threadNum>>> (
        numSequences*4-4,
        numSequences*2,
        d_closest_dis,
        d_closest_id,
        d_closest_branches,
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
        updateClosestNodesBME <<<1,1>>> (
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            d_closest_branches,
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
        
        calculateBranchLengthBME <<<blockNum,threadNum>>> (
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
            d_closest_branches,
            numSequences
        );
        
        auto iter=thrust::min_element(minPos.begin(),minPos.end(),compare_tuple_BME());
        thrust::tuple<int,double,double,double> smallest=*iter;
        // /* print top 5 sorted elements */
        // for(int j=0;j<5;j++){
        //     thrust::tuple<int,double,double,double> s=*iter;
        //     std::cerr<<thrust::get<0>(s)<<" "<<thrust::get<1>(s)<<" "<<thrust::get<2>(s)<<" "<<thrust::get<3>(s)<<'\n';
        //     iter++;
        // } 
        thrust::tuple<int,double,double,double> s = smallest;
        std::cerr<<thrust::get<0>(s)<<" "<<thrust::get<1>(s)<<" "<<thrust::get<2>(s)<<" "<<thrust::get<3>(s)<<'\n';
        
        /*
        Update Tree (and assign closest nodes to newly added nodes)
        */
        int eid=thrust::get<0>(smallest);
        double fracLen=thrust::get<1>(smallest),addLen=thrust::get<2>(smallest);
        updateTreeStructureBME <<<1,1>>>(
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            d_closest_branches,
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
        updateClosestNodesBME <<<1,1>>> (
            d_head,
            d_nxt,
            d_e,
            d_len,
            d_closest_dis,
            d_closest_id,
            d_closest_branches,
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