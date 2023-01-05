# Server-Client Simulations

## MNIST Classifier Instructions
- Run ```python experiment.py -dataset mnist -experiment fedsem```
- ```experiment.py``` supports these tags:
    - ```-dataset```: name of dataset, only mnist
    - ```-experiment```: name of experiment to perform, including fedsem and fedavg
- Edit ```job.yaml``` in ```configs``` directory to modify the runtime settings 