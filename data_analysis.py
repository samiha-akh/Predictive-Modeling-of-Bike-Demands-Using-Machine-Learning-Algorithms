import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import warnings


def get_feature_class(df):
    # Separate numerical and categorical features
    df['snow'] = df['snow'].astype('float64') # all zeros
    categorical = df.select_dtypes(include=['object', 'int64']).columns
    numerical= df.select_dtypes(include=['float64']).columns

    print("Categorical Columns:", categorical.tolist())
    print("Numerical Columns:", numerical.tolist())

def hours_and_months_analysis(df):
    # Group by hour_of_day and calculate counts for increase_stock
    hourly_demand = df.groupby('hour_of_day')['increase_stock'].value_counts(normalize=True).unstack()

    # Group by month and calculate counts for increase_stock
    monthly_demand = df.groupby('month')['increase_stock'].value_counts(normalize=True).unstack()

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))

    # Plot the distribution of bike demand across hours of the day
    hourly_demand.plot(kind='bar', stacked=True, ax=axes[0], color=['skyblue', 'orange'])
    axes[0].set_title("Bike Demand by Time of Day")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Proportion of Demand")
    axes[0].legend(title="Demand Level")

    # Plot the distribution of bike demand across months
    monthly_demand.plot(kind='bar', stacked=True, ax=axes[1], color=['skyblue', 'orange'])
    axes[1].set_title("Bike Demand by Month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Proportion of Demand")
    axes[1].legend(title="Demand Level")

    # Show plot
    plt.tight_layout()
    plt.savefig('hours_and_months.png')
    plt.show()

def hour_day_month_pairplot(df):
    subset = df[['hour_of_day', 'day_of_week', 'month', 'increase_stock']]
    sns.pairplot(subset, hue="increase_stock")
    plt.savefig('hours_days_months_pairplot.png')
    plt.show()

def weekends_and_holiday_analysis(df):
    # Group by 'weekday' and calculate counts for increase_stock
    weekday_demand = df.groupby('weekday')['increase_stock'].value_counts(normalize=True).unstack()

    # Group by 'holiday' and calculate counts for increase_stock
    holiday_demand = df.groupby('holiday')['increase_stock'].value_counts(normalize=True).unstack()

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the distribution of bike demand on weekdays vs weekends
    weekday_demand.plot(kind='bar', stacked=True, ax=axes[0], color=['skyblue', 'orange'])
    axes[0].set_title("Bike Demand: Weekdays vs. Weekends")
    axes[0].set_xlabel("Day Type (0 = Weekend, 1 = Weekday)")
    axes[0].set_ylabel("Proportion of Demand")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Weekend', 'Weekday'])
    axes[0].legend(title="Demand Level", loc='upper right') 

    # Plot the distribution of bike demand on holidays vs non-holidays
    holiday_demand.plot(kind='bar', stacked=True, ax=axes[1], color=['skyblue', 'orange'])
    axes[1].set_title("Bike Demand: Holidays vs. Non-Holidays")
    axes[1].set_xlabel("Holiday (0 = No Holiday, 1 = Holiday)")
    axes[1].set_ylabel("Proportion of Demand")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Holiday', 'Holiday'])
    axes[1].legend(title="Demand Level", loc='upper right') 

    # Adjust the layout for better visualization
    plt.tight_layout()
    plt.savefig('weekdays_holidays.png')
    plt.show()

def temperature_analysis(df):
    # Create bins for temperature (you can adjust the bin size as needed)
    bins_temp = [-30, 0, 10, 12, 15, 17, 20, 30, 50]
    labels_temp = ['<-0°C', '0-10°C', '10-12°C', '12-15°C', '15-17°C', '17-20°C', '20-30°C', '>30°C']

    df['temp_bin'] = pd.cut(df['temp'], bins=bins_temp, labels=labels_temp)


    # Group by temp_bin and calculate counts for increase_stock
    temp_demand = df.groupby('temp_bin')['increase_stock'].value_counts(normalize=True).unstack()

    # Plot the distribution of bike demand for temperature bins
    temp_demand.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
    plt.title("Bike Demand vs. Temperature")
    plt.xlabel("Temperature Range (°C)")
    plt.ylabel("Proportion of Demand")
    plt.xticks(rotation=0)
    plt.legend(title="Demand Level")
    plt.savefig("temperature.png")
    plt.show()

def snowdepth_analysis(df):
    # Create bins and labels to separate 0 snowdepth and non-zero snowdepth
    bins_snow = [-1, 0, df['snowdepth'].max()]  # Use -1 to include 0 in the first bin
    labels_snow = ['0', '> 0']
    df['snow_bin'] = pd.cut(df['snowdepth'], bins=bins_snow, labels=labels_snow)

    # Group by snow_bin and calculate counts for increase_stock
    snow_demand = df.groupby('snow_bin')['increase_stock'].value_counts(normalize=True).unstack()

    # Plot the distribution of bike demand for snowdepth bins
    snow_demand.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
    plt.title("Bike Demand vs. Snowdepth (0 vs. >0)")
    plt.xlabel("Snowdepth")
    plt.ylabel("Proportion of Demand")
    plt.xticks(rotation=0)
    plt.legend(title="Demand Level")
    plt.savefig('snowdepth.png')
    plt.show()

def rain_analysis(df):
    # Create bins for precipitation: 0 and > 0
    bins_precip = [-1, 0, df['precip'].max()]  # Use -1 to include 0 in the first bin
    labels_precip = ['0', '> 0']
    df['precip_bin'] = pd.cut(df['precip'], bins=bins_precip, labels=labels_precip)

    # Group by precip_bin and calculate counts for increase_stock
    precip_demand = df.groupby('precip_bin')['increase_stock'].value_counts(normalize=True).unstack()

    # Create bins for humidity with your specified ranges
    bins_humidity = [0, 20, 40, 60, 80, 100]  # Adjusted bins to cover the range of humidity values
    labels_humidity = ['0 to 20%', '20 to 40%', '40 to 60%', '60 to 80%', '80 to 100%']
    df['humidity_bin'] = pd.cut(df['humidity'], bins=bins_humidity, labels=labels_humidity, include_lowest=True)

    # Group by humidity_bin and calculate counts for increase_stock
    humidity_demand = df.groupby('humidity_bin')['increase_stock'].value_counts(normalize=True).unstack()

    # Plot the graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Precipitation plot
    precip_demand.plot(
        kind='bar',
        stacked=True,
        color=['skyblue', 'orange'],
        ax=axes[0]
    )
    axes[0].set_title("Bike Demand vs. Precipitation (0 vs. >0)")
    axes[0].set_xlabel("Precipitation")
    axes[0].set_ylabel("Proportion of Demand")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Humidity plot
    humidity_demand.plot(
        kind='bar',
        stacked=True,
        color=['skyblue', 'orange'],
        ax=axes[1]
    )
    axes[1].set_title("Bike Demand vs. Humidity")
    axes[1].set_xlabel("Humidity Range")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    # Adjust layout
    fig.suptitle("Bike Demand vs. Precipitation and Humidity", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('rain_analysis.png')
    plt.show()

def main():

    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Load the dataset
    df = pd.read_csv("training_data_fall2024.csv")

    get_feature_class(df)

    hours_and_months_analysis(df)

    hour_day_month_pairplot(df)

    weekends_and_holiday_analysis(df)

    temperature_analysis(df)

    snowdepth_analysis(df)

    rain_analysis(df)




if __name__ == "__main__":
    main()