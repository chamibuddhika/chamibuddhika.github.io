---
layout: post
title: Optimising Matrix Multiplication Using Top Down Analysis
toc: true 
categories: [Performance Engineering, Performance Optimization]
excerpt: In this article I am going to look in to how we can use Top Down microarchitectural analysis method to drive some typical optimizations for matrix multiplication. 
---

## Introduction

Matrix multiplication (in general `dgemm` in BLAS) is one of the performance critical compute 
kernels operating at the heart of deep learning workloads. In this article I am going to 
look in to how we can observe and improve the performance characteristics of a matrix 
multiplication kernel starting from a naive implementation on x86. I am going to illustrate the 
use of Top Down performance analysis, a performance analysis method based on Intel PMU events. 
I have covered Top Down in a previous article which also contains many good references on how
to get started with it. The idea here is to illustrate how we can use this method to drive 
optimizations, using matrix multiplication as a case study. Beating existing highly tuned 
`dgemm` kernels is not a goal of this article. 

## Setup

The hardware is i7, 64GB memory, running Linux xyz with hyperthreading on. 
The code is compiled with g++ version xyz at -O3 optimization level.

## Optimizations

I start with the naive implementation featuring three nested loops as below. Going forward I
use this implementation as the base line.

[Figure]

When run with Top Down enabled, it gives the following output. 

Implementation is severely memory bound as evident from XYZ entries. The matrices are in row-major
order. So every access to matrix B going down the column is going to incur a high amount of last level 
cache misses in the inner-most loop. Only one element of a fetched cache line of B is used in the 
computation at a time and then potentially tossed off the cache which will be refeteched when computing
the next row adajcent cell of C (going right). 

![Naive Implementation](2020-7-27-naive.jpg)

We need to optimise the memory access pattern here in order to maximise the amount of computation done
before relinquishing a fetched cache line from B.

### Tiling (Blocking)

With matrix tiling (aka blocking) we fetch block of each matrix at a time and carry out multiplication
at block level. The fetched blocks are mostly served from the fast memory (caches) and we reduce
the number of main memory accesses incured per floating point operation.

[Figure]

The code now features six nested loops with three inner loops operating at block level.

With this optimization we have reduced the run-time by 3X. Top down output confirms that we
have reduced the last level cache misses and resultant instruction stalls as evident from 
reduced CPI (x% reduction). 

