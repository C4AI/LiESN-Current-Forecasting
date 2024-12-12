# LiESN-Current-Forecasting
Repository for the paper "Improving Current Forecast by Leveraging Measured Data and Numerical Models via LiESNs".

This repository deals with Leaky-Integrator Echo State Networks (LiESN) for forecasting time series. The implemented LiESN uses a varying integration time step to accomodate for irregular time series.

The provided DataLoader receives any number of .csv files with the following columns required:

-datetime: the date and time of the respective row values
-available_since: date and time that indicates when the row is available for use in the model. Numerical models used as input can have available since dates before the datetime of the respective row, since they predict values in the future. Measured values might have available since dates after the datetime if the data takes some time to be available. The LiESN forecast is made using only data available before the forecast start.

Two examples of a single reservoir LiESN and a deep LiESN with two inputs is provided, as well as the files used to optmize the hyperparameters and train the 3 architectures shown in the paper.

The project can be run via docker, following the provided dockerfile, or opened in an IDE. Required Python libraries are listed in the "requirements" file. When running from an IDE, PyTorch with CUDA support needs to be installed manually following the instructions in https://pytorch.org/get-started/locally/