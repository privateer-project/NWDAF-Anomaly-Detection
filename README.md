# NWDAF-Anomaly-Detection
T3.2 Trustworthy AI model building [Anomaly Detection]

## Prerequisites

Before proceeding, ensure you have Docker installed on your system. If you intend to utilize GPU resources, ensure you have the NVIDIA drivers and NVIDIA Docker installed. For detailed instructions on installing NVIDIA Docker, please visit [NVIDIA Docker GitHub](https://github.com/NVIDIA/nvidia-docker).

## Building the Docker Image

To build the Docker image for this project, navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t privateer-ad .
```

## Run the Docker Image
```bash
docker run --gpus all -p 8889:8888 privateer-ad
```

## Accessing the Jupyter Server

Once your Docker container is up and running, you can start interacting with the Jupyter Notebook server. To access the server:

1. **Open your web browser**

2. **Navigate to the Notebook interface**: Enter the following URL in your browser's address bar: http://localhost:8889


## Note regarding NWDAF files
For efficiency and to minimize the Docker image size, certain files and directories are excluded from being transferred into the Docker container.  
.dockerignore configuration excludes all contents of the Data directory except for summary_df.csv, which holds information about the UEs (User Equipment).  
  
If you prefer to include additional files or directories in your Docker build, you can modify the .dockerignore file accordingly. If direct file inclusion is not preferred, you may upload the files through the Jupyter interface once the container is running.