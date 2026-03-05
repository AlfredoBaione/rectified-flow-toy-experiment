# Rectified Flow Toy Experiments

This repository contains the code for **toy experiments with Rectified Flows**, implementing a diffusion-like model to transform 2D distributions. The goal is to learn a velocity field that transports points from a source distribution to a target distribution using rectified flows.

## 🧠 Overview

- **Source distributions:** 2D toy datasets (moons, circles, etc.)  
- **Model:** `ToyModel` – a small neural network predicting velocity fields conditioned on time  
- **Sampling:** Euler integration over steps to move points from source to target  
- **Visualization:**  
  - Step-by-step trajectories with interactive slider  
  - Static snapshots of selected steps  


