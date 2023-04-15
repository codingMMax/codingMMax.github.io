---
title: CUDA based mmWave Radar Object Detection acceleration
date: 2023-03-30 dont know what time it is
categories: [Hardware Acceleration, projects]
tags: [CUDA]
---
## Summary

We are aiming to accelerate a object detection algorithm using CUDA to achieve pipelined low-latency real-time processing. We will also try to accelerate this application using OMP for benchmarking and evaluation.  

## Background

mmWave radar is a widely applied device in modern moving object detection for autonomous driving application. At the same time, GPU or CUDA programming provides outstanding throughput and parallelism in modern computational intensive applications. This moving object detection algorithm relays heavily on techniques from signal processing and linear algebra. The important mathematic techniques such as FFT, Dense Matrix Multiplication are all computing-bound applications that has the potential to have better parallism and achieve better performance in GPU platforms.

The whole system will compute 3 parameters, including the current moving speed of object, the distance between object and detector and angle between object to detector. All these 3 parameter computation can be done independently, which also benifited from the data parallelism.

## The Challenge

There are mainly 3 major difficulties in this project.

- The data dependencies in FFT and pre-processing stages. In this project, the input data will be sliced into several parts and perform FFT twice in sequence but in a different order. This sequential FFT kernel bottlenecks the whole system performance, which may be further exploited to achieve higher parallelism and better data dependencies.
- The data migrating may cause higher latency than expected. Compare to normal ASIC or CPU platforms, the GPU platform has better throughput and computing resources, but the memory / data migrating will much higher.  At the same time, the real-time processing is extreme sensitive to the system latency, which requires us to come out with a precise and tactic design in our system data-path.
- We need to find a large enough data set to guarantee the pipelined system will last enough time to get stable and trustworthy measurements.

## Required Resources

- CUDA programming environment
- RTX high-end GPUs (likely RTX-2080 is enough)
- An NV-link enabled cluster if possible
- Multi-CPU computing cluster (16-processors or more) 

## Goals & Deliverables

#### Required

- Fully functional CUDA accelerated algorithm kernels for speed detection, angle detection and distance detection.
- Well-implemented OMP/MPI accelerated processing algorithms in CPU platforms.

#### Preferred

-  A 10x speedup in CUDA platform comparing to normal CPU based algorithm computing. 
- A GUI window which could plot object trajectories including speed, distance and angles.
- A detailed exploration / explanation states how to parallizing the FFT algorithm in CUDA platform.

## Why this Platform

RTX GPUs are the popular computing platforms in GPU hardware acceleration both for the CUDA programmability as-well-as the great performance outcomes. At the same time, we gain valuable CUDA programming experiences in this course, using RTX GPUs is a natural choice.

## Expected Schueldue

|     Week Count      | Expected Work Content                                        |
| :-----------------: | ------------------------------------------------------------ |
| Week 1 (4.2 - 4.8)  | - Implementing the detection algorithm in CPU version for verification <br>- Implementing the detection algorithm  in CUDA kernel |
| Week2 (4.9 - 4.15)  | - Building pipelined data path for data migrating in both CPU and GPU platform <br>- Implementing the real-time processing code for both CPU and GPU platform |
| Week3 (4.16 - 4.22) | - Evaluating the performance in both CPU and GPU platforms <br>- Further exploration in kernel acceleration including improve the performance of FFT kernel and matrix matrix multiplication kernel |
|  Week4 (4.23-4.29)  | - Further benchmarking CPU platform and GPU platform results including fine-tuning parameters (pipeline stage, core counts and CUDA streaming), re-thinking data migrating path. <br>- Quantitative evaluation the bottlenecks of both CPU and GPU platforms including runing-time cost, data migrating cost, algorithm complexity. |
| Week5 (4.30 - 5.4)  | - Writing final project report <br>- Preparing for the post presentation session |

​    

