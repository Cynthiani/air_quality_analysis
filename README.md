# Report on Air Quality Gas Multisensory Dataset
### Introduction
Air Quality dataset is a time-series dataset that contains an hourly average response from 5 metal Oxide chemical Sensor in an air quality chemical multisensory device. This was deployed in an urban area in an Italian city that is going through pollution problem. The Data spans from March 2004 to February 2025 (1 year) with 9358 data points and 15 features. The dataset also provides hourly average gas concentrations for CO, NMHC, Benzene, NOx, and NO2 from a certified reference analyzer

#### Time Span
March 2004-February 2005 (1 year)
#### Dataset Characteristics
Time-Series
#### Number of Data points /Features
9358 (hourly Averages) / 15 Features
#### Missing Values
Yes; NAN and -200
#### Data Source
UC Irvine Machine Learning Repository
https://archive.ics.uci.edu/dataset/360/air+quality
#### Possible Use Case
Time-series forecasting  
Studying Sensor Drift over Time  
Developing Machine learning models to predict Air quality

#### Features
Air quality dataset contains 15 features as explained below: 
The Air Quality dataset contains the following features:
### Dataset Features  

The Air Quality dataset contains the following features:  

| **Variable Name**    | **Description**                                                                                     | **Unit**   |  
|-----------------------|-----------------------------------------------------------------------------------------------------|------------|  
| **Date**             | Date of record.                                                                                     | -          |  
| **Time**             | Hour of observation.                                                                                | -          |  
| **CO (GT)**          | True hourly averaged concentration of CO (reference analyzer).                                       | mg/m³      |  
| **PT08.S1 (CO)**     | Hourly averaged sensor response (nominally CO targeted).                                             | -          |  
| **NMHC (GT)**        | True hourly averaged overall Non-Methane Hydrocarbon concentration (reference analyzer).             | µg/m³      |  
| **C6H6 (GT)**        | True hourly averaged Benzene concentration (reference analyzer).                                     | µg/m³      |  
| **PT08.S2 (NMHC)**   | Hourly averaged sensor response (nominally NMHC targeted).                                           | ppb        |  
| **NOx (GT)**         | True hourly averaged NOx concentration (reference analyzer).                                         | µg/m³      |  
| **PT08.S3 (NOx)**    | Hourly averaged sensor response (nominally NOx targeted).                                            | -          |  
| **NO2 (GT)**         | True hourly averaged NO2 concentration (reference analyzer).                                         | µg/m³      |  
| **PT08.S4 (NO2)**    | Hourly averaged sensor response (nominally NO2 targeted).                                            | -          |  
| **PT08.S5 (O3)**     | Hourly averaged sensor response (nominally O3 targeted).                                             | -          |  
| **T**                | Temperature.                                                                                        | °C         |  
| **RH**               | Relative Humidity.                                                                                  | %          |  
| **AH**               | Absolute Humidity.                                                                                  | -          |  

### Data Preprocessing and Exploration
The dataset contained missing figures and required some preprocessing. After handling missing values, the dataset went from 9358/15 to 827/14 from which I selected only 1 feature
 ### Model Architecture 
The LSTM model was implemented using Keras and TensorFlow.   
The following is the architecture of the model as:   
Input Shape: (60, 1) where 60 represents the sequence length (timesteps).   
Layers: LSTM  
 Layer 1: Output shape of (60, 50) with 50 units and a dropout rate of 0.2.   
 Dropout Layer 1: Applied after the first LSTM to prevent overfitting.  
  LSTM Layer 2: Output shape of (60, 50) with 50 units.   
  Dropout Layer 2: Applied after the second LSTM layer.  
   LSTM Layer 3: Reduces dimensionality to (50).   
   Dense Output Layer: A fully connected dense layer with 1 unit using Adam optimizer and loss function Mean Squared Error (MSE).   
   
##### Below is a table that highlights the design of the LSTM model and its parameters

### Model Parameters and Details  

| **Parameter**                         | **Detail**                 |  
|---------------------------------------|----------------------------|  
| **Data size after preprocessing**     | 827                        |  
| **Number of Features**                | 1                          |  
| **Testing Percentage**                | 10%                        |  
| **LSTM Layers**                       | 3                          |  
| **Neurons per LSTM Layer**            | 50                         |  
| **Dropout Layers**                    | 2                          |  
| **Dropout Rate**                      | 0.2                        |  
| **Sequence Length**                   | 60-time step               |  
| **Batch Sizes for Training/Testing**  | 32 and 1                   |  
| **Dense**                             | 1                          |  
| **Epochs**                            | 100                        |  
| **Train/Test Split**                  | 90/10 split                |  

 
Fig 14. Designed architecture of our model (designed with Microsoft Visio)
![Designed architecture of the LSTM model](images/lstm_model_architecture.png)







Compiled Architecture design by the model using Adam as optimizer and Mse for loss
  
 Fig 15. Model Architecture

Training Process for the 90% of air quality time series dataset
The model was trained using the time series generator for 100 epochs with a final loss on Epoch 100 as 0.0213. The training loss decreased consistently across epoch, showing effective learning of trends and patterns within the data.  
 
Performance Visualization 
 
Fig 16. Loss curve
The gradual and consistent decrease in the loss curve indicates that there are no major issues with overfitting.
Testing and predicting the dataset with 10% of the Air quality dataset
Forecast Visualization
The predictions as shown below are used to assess the model's capacity to replicate patterns in the test data. The results were charted alongside the training data and the actual closing prices from the test set. The actual data is in blue while the predicted is in orange. The model predicted the data set almost correctly as seen in the figures below
 
Fig 17. Testing and predicted dataset side by side
 
Figure 18 below represents the provided insights into the correlation between the model's predicted and actual values. It is important to note that by superimposing the predicted values with the actual closing prices, it became clear that the model did not perform so well in approximating future trends. I believe that the model can do better with more effective hyperparameter settings to really capture the overall trends and patterns in the test data better. 


 
Fig 18. Co-Joined Testing and predicted dataset 

Model Evaluation and Forecasting using the whole dataset as the training set and predicting a new record for 200hrs 
The entire air quality dataset of 827 points was utilized to train the model for forecasting completely new future values. To achieve the prediction of a future record, the whole dataset was used as training set and extra 200 data points were predicted using the whole trained dataset. I predicted from 5/01/2004 01:00:00 - 05/09/2004 09:00:00 (200 hrs. forecast), it is important to note that my time-series data is an hourly data collection. 
This allowed for evaluation beyond the training dataset. The test set was normalized as the model used to forecast future data points. These predictions were later reverted to their original scale using the inverse of the normalization process applied during data preparation.
To prepare the data, a MinMaxScaler was applied to normalize the complete input dataset, ensuring all values fell within a 0 to 1 range, thus standardizing the data.

Training the Model for Forecasting
The LSTM model was trained on the complete air dataset as visualized in figure 19 below without using a validation set or early stopping mechanisms. This was chosen because the goal of forecasting is to predict unknown future values rather than assess performance on a predetermined test set.
 
Fig 19. Air dataset (whole)

During the training process, the following key parameters were used:
For the Epochs, the model used 100 iterations to adequately minimize the loss function and the batch size of 32 was used as well.  The model demonstrated effective convergence, achieving a low loss value, indicating its capability to accurately model the underlying time series data. 

Predicting Future Values
The trained model (with air dataset) was then used to project future values for 200 hours (5/01/2004 01:00:00 - 05/09/2004 09:00:00). This process involved generating sequential one-step predictions and updating the input sequence by incorporating the predicted value while removing the oldest time step. This rolling forecast method ensured the model remained aware of the most recent trends.
The projected values, initially normalized, were converted back to their original scale using the MinMaxScaler's inverse transformation.
To showcase the forecasting outcomes, the projected values were graphed below. 

 
Fig 20. The Predicted new record (200 hrs.)

Co-Joining the dataset and the newly predicted, this juxtaposition demonstrated how effectively the model captured and extended trends into the future as seen below.


 
Fig 21. Predicted data and actual data together 

Figure 21 above shows a co-joining of historical data (depicted by a blue line) and LSTM model predictions (shown in orange) over a 200-hrs. data point. The blue line illustrates past trends, including variations, high points, and a recent downturn, which the model successfully incorporates and projects in its forecast. 
The predicted values started at the end of the historical data (blue line), continuing the observed downward trajectory with cyclical variations, suggesting the model's effort to mimic established patterns. However, the forecast displays more pronounced downwards trends and fluctuation, indicating the model's constraints in maintaining the patterns beyond its training set, possibly due to the lack of external factors unknown yet. We will need to see the actual data covering the forecasted period to know for sure how good or poorly our model performed.  
The LSTM-based forecasting model tried to exhibit strong performance in predicting future trends over an extended timeframe, however, seem not to achieve it fully. 
In conclusion, the LSTM model proved effective for time series forecasting, providing valuable insights and practical utility in scenarios requiring long-term predictions.



