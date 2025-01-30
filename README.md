# NeuroKB-DR - Neurodegeneration Knowledge Base Drug Repurposing

## Overview

This project uses the existing NeuroKB for drug repurposing. A trained link prediction model is used to score potential edges between a given disease and all drugs in the knowledge base. The top 250 candidate drugs are saved.

## To Use

1. Update the disease in ```/data/input/disease.txt``` using the disease-id pairs in ```/data/db/disease_id.csv```.
2. Download and set up docker
3. Run the following commands:
```docker-compose build```
```docker-compose up```
4. The candidate drugs will be saved to a file in ```/data/output/```

