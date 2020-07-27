---
layout: post
title: Optimising Matrix Multiplication Using Top Down Analysis 
toc: true
categories: [Performance Engineering, Performance Optimization]
excerpt: In this article I am going to look in to how we can use Top Down microarchitectural
analysis method to drive some typical optimizations for matrix multiplication. 
---

## Introduction

Matrix multiplication (in general `dgemm` in BLAS) is one of the critical compute 
kernels operating at the heart of deep learning workloads. Any slight performance 
enhancement in `dgemm` can potentially lead to an outsized performance improvement 
in the overall workload runtime due to how frequently it is being used within the
overall computation of a workload. In this article I am going to look in to how we
can observe and improve the performance characteristics of a matrix multiplication
kernel on x86 using Top Down, a performance analysis method based on Intel PMU events. 

## Setup

The hardware is i7, 64GB memory, running Linux xyz with hyperthreading on. 
The code is compiled with g++ version xyz at -O3 optimization level.
