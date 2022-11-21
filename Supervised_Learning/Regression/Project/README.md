# SPAIN ELECTRICITY LOAD SHORTFALL, 2022
Project Duration: 1 month<br>
Project Date: May 2022
Project Status: First team project with explore

## **TEAM**

* Humphery Ojo
* Samson Oguntuwase
* Thapelo Mofokeng
* Theresa Koomson
* Ayoola Solanke
* Adewale Abiola

## **OVERVIEW**

The supply of electricity plays a large role in the livelihood of citizens in a country. Electricity, amongst other things, helps us stay connected, keep warm, and feed our families. Therefore there's a need to keep the lights on in order to maintain and improve the standard of living by investing in electricity infrastructure. However, in recent years, there has been evidence that the use of purely non-renewable sources is not sustainable.

The government of Spain is considering an expansion of its renewable energy resource infrastructure investments. As such, they require information on the trends and patterns of the country's renewable sources and fossil fuel energy generation. For this very reason, the government of Spain has been tracking the different energy sources available within the country.

In this project, you are tasked to model the shortfall between the energy generated by means of fossil fuels and various renewable sources - for the country of Spain. The daily shortfall, which will be referred to as the target variable, will be modelled as a function of various city-specific weather features such as `pressure`, `wind speed`, `humidity`, etc.

**Acknowledgements**

The original data was sourced from Kaggle

### **Data Description**

This dataset contains information about the weather conditions in various Spanish cities for the time of 2015-2017. The dataset also has information about the three hourly load shortfalls for the same period. In the context of this problem, the three hourly load shortfall is the difference between the energy generated by means of fossil fuels and renewable sources.

The dataset contains 47 features and 1 target. The features include the time and the city-specific weather variables i.e. the wind speed in the city of Barcelona. In total there is weather data available for 5 cities but not all cities have weather information available for every weather category e.g. we might have wind speed data for Barcelona but not rainfall data whereas we have both rainfall and wind speed information for Valencia.

We have weather data for the following cities of Spain:

* Madrid
* Valencia
* Seville
* Bilbao
* Barcelona
The weather categories in the dataset include:

* wind_speed
* wind_degree
* rain_1h
* rain_3h
* humidity
* clouds_all
* pressure
* snow_3h
* weather_id
* temp_max
* temp

### **File descriptions**

df_train.csv - the training set
df_test.csv - the test set

### **Features**

Below follows a brief description of the features and targets contained in the dataset.

* time: Time at which the data was recorded
* {City Name}_wind_speed: The wind speed at a specific time interval for the named city.
* {City Name}_wind_degree: The strength of the wind for the named city at a specific time interval - expressed as a category.
* {City Name}_rain_1h: A metric expressing the amount of rain that has fallen in the past hour in a particular city.
* {City Name}_rain_3h:A metric expressing the amount of rain that has fallen in the past three hours in a particular city.
* {City Name}_humidity: The level of humidity as measured at the defined time for the specific city mentioned.
* {City Name}_clouds_all: The level of cloud coverage as measured at the specified point in time for the specific city mentioned.
* {City Name}_pressure: The atmospheric pressure for the named city at a specific time interval - expressed as a category.
* {City Name}_snow_3h: A metric expressing the amount of snow that has fallen in the past three hours in a particular city.
* {City Name}_weather_id: A metric used to explain the weather condition of a specific city at a specified time.
* {City Name}_temp_max: The maximum temperature for a specific city at a point in time.
* {City Name}_temp_min: The minimum temperature for a specific city at a point in time.
* {City Name}_temp: The average temperature for a specific city at a point in time.

Target Variable

load_shortfall_3h: The difference between the energy generated by the method of renewable energy sources, such as solar, wind, geothermal, etc., and energy generated with fossil fuels - partitioned in three-hour windows.