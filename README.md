# AttnGAN Inference via Pre-trained Model

## Running server
1. Create a virtual environment
```
apt-get install virtualenv
virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt # long time
```
2. Download the data
```
python prep.py
```
3. Run the server on 0.0.0.0:5000
```
python main.py
```

## Running inference
There are three steps involved.
1. Create the container (optionally choose the cpu or gpu dockerfile: 
   ```
   docker build -t "attngan" -f dockerfile.cpu .
   ``` 
2. Run the container: 
    ```
    docker run -it --name attngan -p 8888:8888 attngan bash
    ```
3. Run the jupyter notebook. 

# Credits
All the code has been borrowed from https://github.com/taoxugit/AttnGAN.
This repo just simplifies the evaluation api into a single Jupyter notebook instead of hosting on Azure.


