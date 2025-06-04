# Run the project

The project in docker should be run with:
- `docker compose up --build`

# Organization

Five services will be launched, 4 in the backend and 1 in the frontend. 

In the backend we will have:
- dataset management -> manage dataset for XAI. This will change in the near future for live predictions without the need to manage datasets
- model management -> manage different models and study their inpact. Again, in the near future we will only use the last model.
- XAI Shap -> service to run SHAP upon the model and data
- XAI Lime -> service to run Lime upon the model and data

In the frontend we will have: 
- xai_frontend -> and angular application configured to use the services

After the initial run in docker, the main interface may be accessed in:
- http://localhost:4200

You can also see service documentation in swagger in the following addresses:
- http://localhost:5000
- http://localhost:5001
- http://localhost:5002
- http://localhost:5003

# Produce explanations

1- Load the data test1.csv and the model model.pt from the Load Data in the web GUI. Click the button Load Data

2- Choose an instance of the dataset from 1 to 100 to produce explanations. (In the future, this will be not needed as we will use model detections to generate explanations). Click button load instance data

3 - Generate graphs over the last button. (Only used to graphs for traditional graphs from Shap and Lime)

# See results

In the web UI in the menu TimeSeries we can see an overview of data point influence towards a decision. we can even compare outputs from Shap and Lime side-by-side

In the web UI menu features we can see a short summary of mean feature influence and standard deviation.

In the web UI menu window we can see a short summary of mean window influence and standard deviation.

With these three strategies we can deliberate over most important data points, most important features and time windows. If all features are relevant for the anomally or just a subset. Also if the anomally is present in all time windows or just some specific ones. This information is traditionally not givem by anomally detection algorithms.