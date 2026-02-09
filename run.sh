#!/bin/bash

python main.py batch \
    --configs \
    configs/BeijingPM10Quality/Xgboost.yaml \
    configs/BeijingPM10Quality/Catboost.yaml \
    configs/BeijingPM10Quality/MLP.yaml \
    configs/BeijingPM10Quality/LSTM.yaml \
    configs/BeijingPM10Quality/ResNet1D.yaml \
    configs/BeijingPM10Quality/Transformer.yaml \
    configs/BeijingPM25Quality/Xgboost.yaml \
    configs/BeijingPM25Quality/Catboost.yaml \
    configs/BeijingPM25Quality/MLP.yaml \
    configs/BeijingPM25Quality/LSTM.yaml \
    configs/BeijingPM25Quality/ResNet1D.yaml \
    configs/BeijingPM25Quality/Transformer.yaml \
    configs/Panasonic18650PFData/Xgboost.yaml \
    configs/Panasonic18650PFData/Catboost.yaml \
    configs/Panasonic18650PFData/MLP.yaml \
    configs/Panasonic18650PFData/LSTM.yaml \
    configs/Panasonic18650PFData/ResNet1D.yaml \
    configs/Panasonic18650PFData/Transformer.yaml \
    configs/PPGDalia/Xgboost.yaml \
    configs/PPGDalia/Catboost.yaml \
    configs/PPGDalia/MLP.yaml \
    configs/PPGDalia/LSTM.yaml \
    configs/PPGDalia/ResNet1D.yaml \
    configs/PPGDalia/Transformer.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/Xgboost.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/Catboost.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/MLP.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/LSTM.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/ResNet1D.yaml \
    configs/REVS/2013_Monterey_Motorsports_Reunion/Transformer.yaml \
    configs/REVS/2013_Targa_Sixty_Six/Xgboost.yaml \
    configs/REVS/2013_Targa_Sixty_Six/Catboost.yaml \
    configs/REVS/2013_Targa_Sixty_Six/MLP.yaml \
    configs/REVS/2013_Targa_Sixty_Six/LSTM.yaml \
    configs/REVS/2013_Targa_Sixty_Six/ResNet1D.yaml \
    configs/REVS/2013_Targa_Sixty_Six/Transformer.yaml \
    configs/REVS/2014_Targa_Sixty_Six/Xgboost.yaml \
    configs/REVS/2014_Targa_Sixty_Six/Catboost.yaml \
    configs/REVS/2014_Targa_Sixty_Six/MLP.yaml \
    configs/REVS/2014_Targa_Sixty_Six/LSTM.yaml \
    configs/REVS/2014_Targa_Sixty_Six/ResNet1D.yaml \
    configs/REVS/2014_Targa_Sixty_Six/Transformer.yaml \
    configs/TennesseeEastmanProcess/Xgboost.yaml \
    configs/TennesseeEastmanProcess/Catboost.yaml \
    configs/TennesseeEastmanProcess/MLP.yaml \
    configs/TennesseeEastmanProcess/LSTM.yaml \
    configs/TennesseeEastmanProcess/ResNet1D.yaml \
    configs/TennesseeEastmanProcess/Transformer.yaml \
    configs/VehicleDynamicsDataset/Xgboost.yaml \
    configs/VehicleDynamicsDataset/Catboost.yaml \
    configs/VehicleDynamicsDataset/MLP.yaml \
    configs/VehicleDynamicsDataset/LSTM.yaml \
    configs/VehicleDynamicsDataset/ResNet1D.yaml \
    configs/VehicleDynamicsDataset/Transformer.yaml \
    --output final_results_doublecheck.csv \

echo "All experiments completed!"