Team_2_YESALPHA

Hoon Yao Hong - FeatureEngineering.py, run.sh, Dockerfle, docker-compose.yml
Muhammad Aslam Bin Mohamad Fazli - EDAPython.py, EDAJupyter.ipynb
Ang Wei Jun - ML_load_set_data.py, ML_train_model.py

Running the ML Pipeline:

Step 1: Prerequisites & Setup
It is highly recommended to run this project inside the provided VS Code DevContainer to ensure a consistent environment.
Once your environment is active, open your terminal and install the required Python libraries and run:
pip3 install -r requirements.txt --break-system-packages

Note: Because the DevContainer uses a managed Linux environment, we include the `--break-system-packages` flag to safely install dependencies directly)

Step 2: Execution Permissions
Before running the pipeline for the very first time on a new machine, you must grant the shell script permission to execute:
chmod +x run.sh

Step 3: Run the Pipeline
To execute the entire workflow, simply run the shell script from the root of the project directory:
./run.sh

# Instruction for how to run the docker environment:
## Step 1: Clone the repository (run in command prompt):
	`git clone https://github.com/hoonyaohon123-glitch/Team_2_YESALPHA/tree/main`
	`cd Team_2_YESALPHA`

## Step 2:Build the container (run in command prompt):
	`docker compose up -d`

## Step 3:If container is being viewed on Visual Studio Code(Dev Container):
	1.  Install Dev Containers on Visual Studio Code
	2.  Open up the project folder in Visual Studio Code
	3.  Click the >< button at the bottom left corner
	4.  Select Attach to Running Container
	5.  Select the docker container that was created
	6.  Click on open folder
	7.  Type /app and hit enter

## Step 4.To stop environment (run in command prompt):
	`docker compose down`

Summary of key Findings:

**Data Quality: ** 
Four of the columns had missing values of which are humidity (about 19%), MetalOxideSensor_Unit2 (about 14%), Ambient Light Level (about 11%) and CO_GasSensor (about 8%). There were no duplicated rows that was detected and there was a significant amount of label noise found e.g. LowActivity, Low_Activity along with mixed casing. They were all standardised into one canonical form. 

**Physically Impossible Values: **
Humidity ranged from -49% to 198% where the valid range is between 0% to 100% and CO2_InfraredSensor contained certain negative values and averaged around 109 ppm when the outdoor baseline is around 420 ppm. Some temperatures had also exceeded 100°C which is impossible indoors unless youre being cooked alive. These were identified as synthetic contamination and we fixed them using median imputation.

**Data Cleaning: **
The activity level and HVAC labels were standardised to one consistent form. All the temperature outliers e.g. 307°C, were replaced with the median, around 20°C as they are physically impossible indoors. Any missing numeric values were imputed with the column median. The Ambient Light Level was filled with the mode. The Negative sensor values in humidity and CO_GasSensor were replaced with the column median as gas concentration cannot be negative (stated above). This is to ensure that the dataset now has zero remaining null values.

**Univariate Analysis: **
The CO_GasSensor is right-skewed where most readings are low but there are still some occasional high spikes which actually bring the average up. This is where we used the log1p transform as it is recommended before using linear models. CO2 sensors are near-normal therefore there is no transformation needed. Activity Level is imbalanced(Low is around 57%, Moderate at around 31% and High at around 12%) which will be addressed using either class weights or SMOTE during the modelling. If the Kurtosis is greater than 1, it means that there are more frequent extreme values than a normal distribution.

**Correlation Analysis: **
The MetalOxideSensor_Unit4, MetalOxideSensor_Unit2 and C02_ElectroChemicalSensor have the highest linear correlation with the activity level. The C02_InfraredSensor and C02_ElectroChemicalSensor are highly correlated with each other and are redundant for linear models. The humidity has the lowest correlation with the target.

**Bivariate Analysis: **
The CO2 sensors are the strongest predictors compared to the others as the readings increase with activity level. The Metal Oxide Sensor Units 1 and 3 show a moderate separation across the different activity classes. Ambient Light Level and Time of Day also show clear patterns where High Activity correlates with brighter environments and morning/afternoon periods.

**Session Analysis: **
The sessions vary in length where some have only a handful of readings while others have dozens. Most of the sessions are dominated by Low Activity readings which is consistent with the overall class distribution. The CO2 levels also vary across sessions reflecting the different residents or monitoring periods. The Session ID should not be used as a predictive feature but is useful for grouped cross-validation to prevent data leakage.

# Feature Engineering(How and why):

### 1. Temporal Dynamics:
		This feature takes the current reading and subtracts it from the previous reading to find the step-by-step difference. This is done so the model can easily detect trends in the data.

### 2. Rolling Window Statistics:
		This feature calculates the standard deviation of the data from its last 5 readings. This measures the stability of the data, giving the model historical context over a short timeframe, allowing it to distinguish between a steady state and a turbulent state.

### 3. Sensor Fusion:
		This features take the 4 Metal Oxide sensors and turns them into two new columns: the mean and the highest reading of the sensors at that exact second. By taking the mean, it reduce dimensionality and noise. While taking the max reading mean that it is capturing the worst-case scenarios as the mean may be diluted.

### 4. Session-level Baselines:
		These features calculate the overall average reading for an entire session then subtract that session average from the current reading. This normalizes the data to handle day-to-day environmental shifts.

### 5. Domain-Specific Features:
		* Features include Human Discomfort Level, Total Airborne Gas Level, Gas Buildup Over Time and Abnormal CO2 Spike warning


		* Human Discomfort Level: Applies Thom's Discomfort Index combining temperature and humidity. This maps raw physical metrics to human physiological comfort so this can help the model if it is trying to predict human activity.


		* Total Airborne Gas Level and Gas Buildup Over Time: Sums up the raw values of all metal oxide sensors, and then calculates a rolling 10-step sum. Represents the total concentration of gases in the air and shows how they pool or clear out over time.


		* Abnormal CO2 Spike warning: A label that triggers when the current CO2 reading is significantly higher than the recent rolling average. 	This acts as an anomaly detector, flagging sudden spikes for the model to detect sudden environment events.

### 6. Text-to-Numeric Translation
		* This features does categorical encoding, turning English words into binary.Machine learning algorithms can only perform mathematical operations on numbers, not English strings.


# Model Selection, Evaluation, Tuning Methods

## 1. Model Pipeline process:

`cd src`
`python ML_load_set_data.py`
`python ML_train_model.py`

> loads, cleans, and uploads feature-engineered data
> generates 3 baseline models (RANDOM FOREST CLASSIFIER, XGBOOST, LOGISTIC REGRESSION), and shows classification report
> loops through a training loop, tunes hyperparameters on RandomisedSearchCV, notes best models and saves all models.


The .py files runs in sequence using functions and Object-Oriented Programming to run commands in sequences to stay organised and keep code reproducible, readable, flexible, and easily modifiable.

## 2. Evaluation Method
	Macro F1 is the deciding factor of the 'best' model in this project because our data is imbalanced generally (Low Activity, Moderate Activity, High Activity). Imbalanced data can cause skewness or bias in the models. Hence we need to treat all classes equally and penalize neglect. 


	> Macro F1 computes the F1 Score for every class independently and then takes the unweighted average F1 score.


	Hence a Macro F1 Score would allow the model to care about the "High Activity" events, even if they are not many of them. Especially in sensor data, ensuring no biasness or imbalances are involved in the Recall or Precision metrics behind the model and data-driven decision making.

## 3. Model selection and tuning methodology

	We used three models to capture the data's complexity and evaluated on all three.

	a. **Logistic Regression** - a simpler model that does not use much resources (compared to the others). We use this as one of the baselines to test if more complex and resource-heavy models are actually adding value.

		> But it does not do well on complex data.

	b. **Random Forest** - a bagging-based ensemble good for capturing non-linear interactions. As there are outliers in the data, such as contaminated/messy data and not-so-well-known relationships between the data, Random forest is a robust model used a lot in the real world for complex data.

		> However, it can be resource heavy and slow to train.

	c. **XGBoost (gradient boosting)** - a model known for parallelized processing and training speed used to implement gradient boosted decision trees to learn from its mistake / previous iteration. It is good with tabular data and good at finding subtle non-linear relationships between data points.

		> However, it is prone to overfitting and also it needs to be carefully tuned.

	Using these pros and cons, the python file will loop through each model and print out a classification report.


## 4. Tuning Strategy
	a. **RandomizedSearchCV**
	b. **5-Fold Cross Validation**

		We utilised RandomizedSearchCV, instead of GridSearchCV, with 5-fold Cross Validation method.


		A 5 Fold CV is used because, while it takes a longer time to compute, a 3 Fold CV is prone to bias since our data is already imbalanced, it may be biased since it only checks through 33% of the data.


		It may seem counterintuitive why we still used a 5 Fold Cross Validation over a 3 Fold one while we already used SMOTE technique on our data to balance it. The reason is because the data SMOTE generated is still synthetic. Hence a 5 Fold Cross Validation would stress test our model more to note when it is overfitting or being biased.


## 5. Balancing Classes

	**Using SMOTE (Synthetic Minority Over-sampling Technique).**
		SMOTE creates synthetic data (nearest neighbours) to the actual data to balance out all the classes by adding them to a minorty class. It lets a model learn the boundary of each class (i.e, what classifies as a High Activity or Low Activity). 
