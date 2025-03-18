# GAMR MLOps Project

This repo is part of the GAMR MLOps project, designed to showcase a flexible and scalable MLOps architecture. By integrating multiple clients and model registries, it simulates real-world industry scenarios where different components need to work together seamlessly. The goal is to demonstrate how a well-structured, decoupled system can make AI deployments more efficient, adaptable, and easy to maintain.

# MLFlow Model Registry and Experiment Tracker

This repository contains the model registry and experiment tracking component for the GAMR MLOps project. It is designed to manage model versions, track experiments, and serve models via API.

## Architecture Overview

**Model Registry**

* MLFlow-based registry for versioning and tracking model performance.

Important! Due to free-tier hosting limitations, the deployed service uses a lightweight, dummy model registry instead of MLFlow’s full server-based solution.

**Experiment Tracking**

* Tracks training runs, parameters, and performance metrics.

## Usage

1. Open the app in [https://gamr-image-recognition.streamlit.app](https://gamr-image-recognition.streamlit.app).

2. Chose either object detection or flower classification.

3. View predictions and results in real-time.

- **For image classification:** Upload an image, and the model will return the predicted class.
- **For iris classification:** Input sepal/petal measurements, and the model will predict the iris species.


## WIP and Future Enhancements
- 100% test coverage.
- Enhanced experiment visualization and comparison tools.

---

Feel free to suggest improvements!
