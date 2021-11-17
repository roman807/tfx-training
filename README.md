# tfx-training

To run the pipeline with Apache Airflow

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


