## Orchestration: A Simple ML Pipeline 

### Initiate Project
This work is a simple ml orchestration tool with the **zenml** architecture for MLOPs - that has the ends with the task of promoting the trained model.

To follow through, you would need a virtual environment, install the required dependencies 
and launch **zenml**! __Python 3.10 or lower is required.__

### SetUp Virtual Environment
```commandline
python3.10 venv -m <venv_name>
<venv_name>\Scripts\activate
```

### Activate Venv
After you must have created a virtual environment, activate it and follow the steps as stated on the 
[zenml site](https://docs.zenml.io/user-guide/starter-guide/create-an-ml-pipeline)

```commandline
pip install "zenml[server]"
zenml integration install sklearn -y
pip install -r requirements.txt
```
### Install Dependencies
Now, it is ready to roll.
We would be working with a single simple ML Pipeline that:
- - *Fetches Data* -> *Train Model* -> *Promote Model*

### Run Pipeline
Execute Pipeline
```commandline
python model_orchestration.py
```
Display User Interface
```
zenml up
zenml up --blocking #Windows
```

[BLOG](https://dev.to/afrologicinsect/zenml-for-beautiful-beautiful-orchestration-46db)
