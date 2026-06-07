Team_2_YESALPHA

Hoon Yao Hong - FeatureEngineering.py


Instruction for how to run the docker environment:
Step 1: Clone the repository (run in command prompt):
	git clone https://github.com/hoonyaohon123-glitch/Team_2_YESALPHA/tree/main
	cd Team_2_YESALPHA

Step 2:Build the container (run in command prompt):
	docker compose up -d

Step 3:If container is being viewed on Visual Studio Code(Dev Container):
	1.  Install Dev Containers on Visual Studio Code
	2.  Open up the project folder in Visual Studio Code
	3.  Click the >< button at the bottom left corner
	4.  Select Attach to Running Container
	5.  Select the docker container that was created
	6.  Click on open folder
	7.  Type /app and hit enter

Step 4.To stop environment (run in command prompt):
	docker compose down



Feature Engineering(How and why):

1. Temporal Dynamics:
	This feature takes the current reading and subtracts it from the previous reading to find the step-by-step difference. This is done so the model can easily detect trends in the data.

2. Rolling Window Statistics:
	This feature calculates the standard deviation of the data from its last 5 readings. This measures the stability of the data, giving the model historical context over a short timeframe, allowing it to distinguish between a steady state and a turbulent state.

3. Sensor Fusion:
	This features take the 4 Metal Oxide sensors and turns them into two new columns: the mean and the highest reading of the sensors at that exact second. By taking the mean, it reduce dimensionality and noise. While taking the max reading mean that it is capturing the worst-case scenarios as the mean may be diluted.

4. Session-level Baselines:
	These features calculate the overall average reading for an entire session then subtract that session average from the current reading. This normalizes the data to handle day-to-day environmental shifts.

5. Domain-Specific Features:
	Features include Human Discomfort Level, Total Airborne Gas Level, Gas Buildup Over Time and Abnormal CO2 Spike warning
	Human Discomfort Level: Applies Thom's Discomfort Index combining temperature and humidity. This maps raw physical metrics to human physiological comfort so this can help the model if it is trying to predict human activity.
	Total Airborne Gas Level and Gas Buildup Over Time: Sums up the raw values of all metal oxide sensors, and then calculates a rolling 10-step sum. Represents the total concentration of gases in the air and shows how they pool or clear out over time.
	Abnormal CO2 Spike warning: A label that triggers when the current CO2 reading is significantly higher than the recent rolling average. This acts as an anomaly detector, flagging sudden spikes for the model to detect sudden environment events.

6. Text-to-Numeric Translation
	This features does categorical encoding, turning English words into binary.Machine learning algorithms can only perform mathematical operations on numbers, not English strings.
