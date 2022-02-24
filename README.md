# Predicting Electric Grid Stability on AzureML

The stability of the electric power grid is very important becuase stabilities can cause damage to the grid components or even black outs.

In this project, I used Azure Hyperdrive and Azure AutoML to predict the stability of a 4-node star system.

The steps followed are:

![Project Diagram](/img/0-ProjectDiagram.png)

## Dataset

### Overview

The dataset "Electrical Grid Stability Simulated Data" was obtained from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+#).

The dataset consists of 10000 observations of an electric power grid with four nodes in total, one with electricity generation and three with electricity consumption:

![Diagram](/img/0-Diagram.png)

The dataset contains the following features (node 1 refers to the electricty producer, whereas nodes 2 to 4 refers to the electricity consumer):

    - tau: reaction time of each participant (1 to 4).
    - p: nominal power consumed or generated of each node (1 to 4). A negative value indicates a net consumption, whereas a positive value indicates net generation.
    - g: coefficient (gamma) proportional to price elasticity (1 to 4).
    - stab: the maximal real part of the characteristic equation root. A positive value indicates that the system is linearly unstable.
    - stabf: the stability label of the system. This is a categorical feature: stable/unstable.

More information can be found in the following papers:
* [Taming Instabilities in Power Grid Networks by Decentralized Control](https://arxiv.org/pdf/1508.02217.pdf)
* [Towards Concise Models of Grid Stability](https://dbis.ipd.kit.edu/download/DSGC_simulations.pdf)

### Task
The aim of this project is to develop a machine learning model that can predict the stability of the system based on the features of the dataset.
Therefore, it is a classification problem in which we want to predict the stabf value.

For this task, we use the features mentioned above except for stab, since we are using its categorical feature stabf.

![Dataset](/img/0-Dataset.png)

### Access

After reading and cleaning the data, I uploaded it to a datastore on the cloud and created a dataset referencing to the cloud location as explained [here](https://stackoverflow.com/questions/60380154/upload-dataframe-as-dataset-in-azure-machine-learning). 

## Automated ML

These are the AutoML settings and configuration that I chose:

```ruby
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy',
    "blocked_models": ['XGBoostClassifier']
}

automl_config = AutoMLConfig(compute_target = compute_target,
                             task = "classification",
                             training_data = ds,
                             label_column_name = "stabf",   
                             path = project_folder,
                             enable_dnn = True,
                             enable_early_stopping = True,
                             featurization = 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

| Parameter  | Description |
| ------------- | ------------- |
| experiment_timeout_minutes  | 20 minutes. This is the maximum number of minutes that the experiment will run for in order to avoid extra costs. |
| max_concurrent_iterations | 5 iterations. This is the maximum number of iterations that will be run in parallel. |
| primary_metric  | accuracy. This is the [metric](https://docs.microsoft.com/en-us/python/api/azureml-automl-core/azureml.automl.core.shared.constants.metric?view=azure-ml-py) that will be used to determine the best model. Accuracy is the proportion of true results among the total number of cases examined and it is used in this case because there is no class imbalance. |
| blocked_models | 'XGBoostClassifier'. This is the list of models that will not be evaluated. In this case, I choose to block the XGBoostClassifier model because to see how this setting works. |
| compute_target | compute_target defined. This is the compute target that will be used to run the experiment. |
| task | 'classification'. This is the task that we want to perform according to our problem. |
| training_data | ds. This is the dataset that we want to use to train our model. It includes the features and the labels. |
| label_column_name | "stabf". This is the label column name that we want to use to train our model. |
| path | project_folder. This is the path to the project folder. |
| enable_dnn | True. This is the flag that indicates if we want to consider using deep neural networks. |
| enable_early_stopping | True. This is the flag that indicates if we want to stop the experiment once the results seem not to improve with new iterations. |
| featurization | "auto". This indicates that the featurization will be automatically performed. |
| debug_log | "automl_errors.log". This is the file where the debug information will be logged. |

### Results

The best performing AutoML model is a an [Ensemble model](https://en.wikipedia.org/wiki/Ensemble_learning) with an accuracy of approximately 94%.

![GetDetails1](/img/1-GetDetails.png)

![RunDetails1](/img/1-RunDetails.png)

## Hyperparameter Tuning

HyperDrive is used to tune the hyperparameters of a Logistic Regression model, which are Regularization Strength (C) and Maximum Number of Iteratations (max_iter). A Logistic Regression is a good choice for this classification problem in order to stablish a baseline.

Random Parameter Sampling randomly selects hyperparameters to evaluate, which makes the hyperparameter tuning convergence fast. 
The sampler chooses between discrete values of C and max_iter.

Bandit Policy is an early termination policy based on slack factor/slack amount and evaluation interval. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

Again, in this problem, we want to maximize the accuracy of the model.

```ruby
ps = RandomParameterSampling({"--C":choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
                             "--max_iter":choice(50,100,200,300)})

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1, delay_evaluation=1)

hd_config = HyperDriveConfig(run_config=src,
                             hyperparameter_sampling=ps,
                             primary_metric_name="Accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             policy=policy,
                             max_total_runs=16,
                             max_concurrent_runs=4)
```

### Results

The Logisitc Regression model after training has an accuracy of approximately 82% and values C = 1 and max_iter = 50.

![Hyperparameters](/img/2-Hyperparameters.png)

![RunDetails2](/img/2-RunDetails.png)

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

After comparing the results, I deployed the model that we obtained using AutoML using an [Azure Container Instance](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aciwebservice?view=azure-ml-py) since it has a higher accuracy.

After the deployment, a REST API endpoint is available. In the Consume tab of the endpoint, there are the REST endpoint and authentication keys, which are used in the endpoint script to make a prediction based on input values. 

![APIDetails](/img/3-APIDetails.png)
![APIConsume](/img/3-APIConsume.png)
![Endpoint](/img/3-Endpoint.png)

After consuming the endpoint, we see that the result is 1, meaning that the grid is unstable based on those conditions. Finally, we can delete the service.

![ConsumptionAndDelete](/img/3-ConsumptionAndDelete.png)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response