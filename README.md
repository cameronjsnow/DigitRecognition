# Handwritten Digit Recognition Web App

A production-ready machine learning web application that recognizes handwritten digits drawn by users in real-time. This project leverages a deep learning model (CNN with PyTorch) for inference, a Flask backend for serving predictions, Tailwind CSS for a sleek UI, a SQLite database for logging predictions, and Docker for easy deployment.

## Features

- **Real-Time Inference:** Draw digits directly on a canvas and get immediate predictions.
- **Deep Learning Model:** Utilizes a Convolutional Neural Network trained on MNIST for high accuracy.
- **Data Persistence:** Stores prediction logs in a SQLite database.
- **API Documentation:** OpenAPI/Swagger endpoint at `/api/docs`.
- **Deployment Ready:** Dockerized for easy deployment. Optionally integrate CI via GitHub Actions.

## Technologies Used

- **Python**: Core language for model inference and server-side logic.
- **Machine Learning (PyTorch)**: CNN model for digit recognition.
- **Flask**: Lightweight web framework for serving predictions.
- **SQLite & SQLAlchemy**: Simple, file-based database for logging predictions.
- **Tailwind CSS**: Clean, minimal styling for a modern and responsive UI.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
