# TFX-training

The TFX training consists of three separate exercises. They build on each others knowledge, but can be executed independently. The sample data used is the [Laptop Prices Prediction dataset](https://www.kaggle.com/danielbethell/laptop-prices-prediction) with the objective to create a workflow for a model that predicts laptop price based on some descriptive attributes.

## Part 1: TFX components
Create, run and inspect the individual components of a TFX pipeline in a Colab notebook. Access the notebook [here](https://colab.research.google.com/drive/1oQHDhYEkXdmUXGApS6E8q1yKSXOODHrv?usp=sharing)

## Part 2: Run pipeline, ML Metadata and TF Serving
In this part you will run the complete pipeline (composed of the components defined in the first exercise) with the LocalDagRunner. Subsequently you will explore the created artifacts with the ML Metadata store and finally serve the model with TF Serving for inference request via REST API. Access the notebook [here](https://colab.research.google.com/drive/1ICOvTiHVIBm-YuV___zbE46vyvSIPfWr?usp=sharing)

## Part 3: Run the complete pipeline with Apache Airflow

This part is executed from the command line. The instructions assume you are working on an UNIX compliant machine (e.g. Linux or Mac).

1. Clone this repo and cd into it
```
git clone https://github.com/roman807/tfx-training.git
cd tfx-training
```

2. Create and activate a Python environment with Python version 3.6. E.g. using conda:
```
conda create --name tfx_laptop python=3.6
conda activate tfx_laptop 
```

3. Install the following libraries:
```
pip install apache-airflow==1.10.13
pip install -U tfx
pip install SQLAlchemy==1.3.15
pip install wtforms==2.3.3
```

4. Initialize airflow
```
airflow initdb
```

5. Configure paths for pipeline run
```
export AIRFLOW_HOME=~/airflow
export LAPTOP_DIR=~/laptop
```

6. Copy dag runner file and pipeline script to airflow dags and data & utils to "laptop"-path
```
mkdir -p $AIRFLOW_HOME/dags/
cp run_pipeline_airflow.py $AIRFLOW_HOME/dags/
cp laptop_pipeline.py $AIRFLOW_HOME/dags/

cp -R data $LAPTOP_DIR
cp -R utils $LAPTOP_DIR
```

7. Start the airflow webserver (e.g. at port 8080)
```
airflow webserver -p 8080
```

8. Open a new terminal, activate the environment (e.g. ```conda activate tfx_laptop```)
```
airflow scheduler
```

9. open the Airflow webapp (```e.g. localhost:8080```), activate the correct DAG ("laptop_pipeline_airflow"), trigger DAG and inspect the run


