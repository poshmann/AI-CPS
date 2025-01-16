# AI-CPS: Nepal Earthquake Severity Prediction

This repository is a fork of the `MarcusGrum/AI-CPS` repository and has been modified to include a custom project for the course:

**“M. Grum: Advanced AI-Based Application Systems”**  
**Junior Chair for Business Information Systems, esp. AI-Based Application Systems**  
**University of Potsdam**

---

## Project Overview
The project aims to predict earthquake severity levels using data from the Nepal Earthquake dataset. This involves:
- Scraping, cleaning, and processing earthquake-related data.
- Developing AI models using TensorFlow.
- Implementing OLS models for comparison.
- Packaging the solution into Docker images for deployment.

---

## Repository Structure
- **`data/`**: Contains raw and processed data files (`joint_data_collection.csv`, `training_data.csv`, etc.).
- **`models/`**: Stores trained AI and OLS models.
- **`code/`**: Python scripts for data preprocessing, model training, and evaluation.
- **`docker/`**: Dockerfiles and related configurations.
- **`documentation/`**: Course-related documentation and the final team report.
- **`images/`**: Example Docker images for the project.
- **`scenarios/`**: Sample `docker-compose.yml` files for integrating AI models and data processing.

---

## Dataset
The dataset used in this project is the **Nepal Earthquake Dataset** from Kaggle, sourced from:
[https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/)

Data files:
- **`joint_data_collection.csv`**: Cleaned and combined dataset.
- **`training_data.csv`**: 80% of the dataset for training.
- **`test_data.csv`**: 20% of the dataset for testing.
- **`activation_data.csv`**: A single data point for model activation testing.

---

## License
This repository adheres to the terms of the **AGPL-3.0 License** as required by the course.

---

## Acknowledgments
This project was created as part of the course “M. Grum: Advanced AI-Based Application Systems” at the University of Potsdam.
