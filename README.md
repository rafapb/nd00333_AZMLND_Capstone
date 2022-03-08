# Predicting Electric Grid Stability on AzureML

The stability of the electric power grid is very important because system instability can cause damage to the grid components or even black-outs.

In this project, I used Azure Hyperdrive and Azure AutoML to predict the stability of a 4-node star system. Then, I deployed the best model and consumed the model endpoint.

The steps followed are:

![Project Diagram](/img/0-ProjectDiagram.png)

## Dataset

### Overview

The dataset "Electrical Grid Stability Simulated Data" was obtained from the UCI Machine Learning Repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+#).

The dataset consists of 10000 observations of an electric power grid with four nodes in total, one with electricity generation (in the middle) and three with electricity consumption:

![Diagram](/img/0-Diagram.png)

The dataset contains the following features (node 1 refers to the electricty producer, whereas nodes 2 to 4 refers to the electricity consumer):

| Feature  | Description |
| ------------- | ------------- |
| tau | Reaction time of each participant (1 to 4). |
| p | Nominal power consumed or generated of each node (1 to 4). A negative value indicates a net consumption, whereas a positive value indicates net generation. |
| g | Coefficient (gamma) proportional to price elasticity (1 to 4). |
| stab | The maximal real part of the characteristic equation root. A positive value indicates that the system is linearly unstable. |
| stabf | The stability label of the system. This is a categorical feature: stable/unstable. |

More information can be found in the following papers:
* [Taming Instabilities in Power Grid Networks by Decentralized Control](https://arxiv.org/pdf/1508.02217.pdf)
* [Towards Concise Models of Grid Stability](https://dbis.ipd.kit.edu/download/DSGC_simulations.pdf)

### Task
The aim of this project is to develop a machine learning model that can predict the stability of the system based on the features of the dataset.
Therefore, it is a classification problem in which we want to predict the stabf value.

For this task, we use the features mentioned above except for stab, since we are using its categorical feature stabf. 
The following image shows the first five observations of the clean dataset: 

![Dataset](/img/0-Dataset.png)

### Access

After reading and cleaning the data, I uploaded it to a datastore on the cloud and created a dataset referencing to the cloud location as explained [here](https://stackoverflow.com/questions/60380154/upload-dataframe-as-dataset-in-azure-machine-learning). 

## Automated ML

These are the [AutoML Settings](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#configure-your-experiment-settings) and [AutoML Config](https://docs.microsoft.com/en-us/python/api/azureml-train-automl-client/azureml.train.automl.automlconfig.automlconfig?view=azure-ml-py) that I chose:

```ruby
automl_settings = {
    "experiment_timeout_minutes": 15,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'
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
                             enable_voting_ensemble=False,
                             enable_stack_ensemble=False,
                             **automl_settings
                            )
```

| Parameter  | Description |
| ------------- | ------------- |
| experiment_timeout_minutes  | 15 minutes. This is the maximum number of minutes that the experiment will run for in order to avoid extra costs. |
| max_concurrent_iterations | 5 iterations. This is the maximum number of iterations that will be run in parallel. |
| primary_metric  | accuracy. This is the [metric](https://docs.microsoft.com/en-us/python/api/azureml-automl-core/azureml.automl.core.shared.constants.metric?view=azure-ml-py) that will be used to determine the best model. Accuracy is the proportion of true results among the total number of cases examined and it is used in this case because there is no class imbalance. |
| compute_target | compute_target defined. This is the compute target that will be used to run the experiment. |
| task | 'classification'. This is the task that we want to perform according to our problem. |
| training_data | ds. This is the dataset that we want to use to train our model. It includes the features and the labels. |
| label_column_name | "stabf". This is the label column name that we want to use to train our model. |
| path | project_folder. This is the path to the project folder. |
| enable_dnn | True. This is the flag that indicates if we want to consider using deep neural networks. |
| enable_early_stopping | True. This is the flag that indicates if we want to stop the experiment once the results seem not to improve with new iterations. |
| featurization | "auto". This indicates that the featurization will be automatically performed. |
| debug_log | "automl_errors.log". This is the file where the debug information will be logged. |
| enable_voting_ensemble | False. This is the flag that enables or disables using a voting ensemble model. |
| enable_stack_ensemble | False. This is the flag that enables or disables using a stack ensemble model. |

### Results

The best performing AutoML model is a a [LightGBM](https://en.wikipedia.org/wiki/LightGBM) with an accuracy of approximately 94%.

The following image shows a screenshot of the details of the best run:

![GetDetailsAutoML](/img/1-GetDetails.png)

The following image shows a screenshot of the RunDetials widget of the AutoML run:

![RunDetailsAutoML](/img/1-RunDetails.png)

The following image shows a screenshot of the details of the best model, including the model's hyperparameters:

![BestModelAutoML](/img/1-FitModel.png)

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

The following image shows a screenshot of the hyperparameters of the best model:

![Hyperparameters](/img/2-Hyperparameters.png)

The following image shows a screenshot of the RunDetials widget of the HyperDrive run:

![RunDetailsHD](/img/2-RunDetails.png)

## Model Deployment

After comparing the results, I [deployed](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python) the model that we obtained using AutoML using an [Azure Container Instance (ACI)](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.aciwebservice?view=azure-ml-py) since it has a higher accuracy.

After the deployment, a REST API endpoint is available. In the Consume tab of the endpoint, there are the REST endpoint and authentication keys, which are used in the endpoint script to make a prediction based on input values. 

The following image shows the ACI web service configuration:

![ACI](/img/3-ACI.png)

We download the scoring file that AutoML created automatically and use it as our entry script together with our environment to set up our inference configuration.
We configure the endpoint deployment configuration.
We deploy the model and then it is ready to be consumed.

More information can be found [here](https://stackoverflow.com/questions/66437607/how-to-create-azure-ml-inference-config-and-deployment-config-class-objects-from) and [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python).

```ruby
service_name = 'gridstability-api'
script_file_name = 'score.py'

best_run.download_file('outputs/scoring_file_v_1_0_0.py', script_file_name)
best_run_env = best_run.get_environment()
inference_config = InferenceConfig(entry_script=script_file_name, environment=best_run_env)

aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True, enable_app_insights=True)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
```

In Endpoints, we can see that the gridstability-api is healthy:

![APIDetails](/img/3-APIDetails.png)

In the Consume tab of the endpoint, we can see the REST endpoint and authentication keys:

![APIConsume](/img/3-APIConsume.png)

We copy the REST endpoint and primary authentication key to the score.py script, where we also include some sample data:

![EndpointScript](/img/3-EndpointScript.png)

Then, we run the endpoint script. After consuming the endpoint, we see that the result is 1, meaning that the grid is unstable based on those conditions:

![ConsumptionAndDelete](/img/3-ConsumptionAndDelete.png)

Finally, we can delete the service to avoid incurring in extra costs.

## Screen Recording

The screencast of this project can be found [here](https://www.youtube.com/watch?v=xT_gqpj66kU).

## Future Work

* Export the model to support [ONNX](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx).
* Retrain the model periodically with more recent data and [monitor data drift](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?tabs=python).
* Repeat this process on data from more complex electric power grids with more nodes, prosumers, energy storage, etc. 
