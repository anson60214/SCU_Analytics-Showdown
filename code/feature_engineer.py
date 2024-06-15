# Common imports
import numpy as np 
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans

class FeatureEngineer:

    def __init__(self):
        # Initialize any parameters or attributes needed for feature engineering
        pass

    target = {"target1":"default_ratio", 
              "target2":"default"}

    numerical_feature = [
        "down_payment",
        "down_payment_days_included",
        "minimum_payment",
        "full_price",
        "payment_amount_per_period",
        "payment_period_in_days",
        "nominal_term_days",
        "total_paid",
        "mean_income",
        "age",
        "num_payment_term",
        "max_payment_list",
        "mean_payment_list"
        ]

    categorical_feature = [
        "product_size",
        "state",
        "gender",
        "occupation",
        "business_type",
        "business_size",
        "lead_source",
        "product_use",
        "registration_date_year",
        "registration_date_month",
        "registration_date_weekday",
        "latest_payment_date_year",
        "latest_payment_date_month",
        "latest_payment_date_weekday",
        "geo_cluster"
        ]
    
    def extract_PAYG(self, df:pd.DataFrame):
        return df[df['billing_model']=="PAYG"]
    
    def regist_time_extraction(self, df:pd.DataFrame, time_col:str):
        df = df.copy()
        
        # Convert 'registration_date' to datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Extract year, month, and weekday
        df[time_col+'_year'] = df[time_col].dt.year
        df[time_col+'_month'] = df[time_col].dt.month
        df[time_col+'_weekday'] = df[time_col].dt.day_name() 
        
        # Return the modified DataFrame
        return df
    
    def long_lat(self, df:pd.DataFrame):
        df= df.copy()

        # Split 'geolocation' into 'latitude' and 'longitude'
        df[['latitude', 'longitude']] = df['geolocation'].str.split(',', expand=True)
        # Convert 'latitude' and 'longitude' to float
        df['latitude'] = pd.to_numeric(df['latitude'])
        df['longitude'] = pd.to_numeric(df['longitude'])
        X= df[df[['latitude', 'longitude']]\
                        .notna().all(axis=1)][['latitude', 'longitude']]

        X = X[X["latitude"]<=20]
        X = X[X["longitude"]<=25]

        # Kmean for cluster the geo
        km = KMeans(n_clusters=10, random_state=0)
        km.fit(X)

        #plt.scatter(X['longitude'], X['latitude'], c=km.labels_)

        X["geo_cluster"] = km.labels_

        df = df.merge(X, left_index=True,right_index=True, how='left')
        df = df.drop(["latitude_y","longitude_y"],axis=1)
        df = df.rename(columns={"latitude_x":"latitude","longitude_x":"longitude"})
        df.loc[df['geolocation'].notna() & df['geo_cluster'].isna(),"geo_cluster"] =  10

        # Using pandas factorize
        df["geo_cluster"] = df["geo_cluster"].astype(str)
        return df
    
    def mean_income(self, df:pd.DataFrame):
        df = df.copy()

        # Function to calculate the mean income from the range string
        def calculate_mean(range_str):
            if range_str == "50,000 and Below":
                return 25000
            elif range_str == "1,000,001 and Above":
                return 2000000
            else:
                if pd.notna(range_str):
                    lhs, rhs = range_str.replace(',', '').split(' – ')
                    lhs = int(lhs)
                    rhs = int(rhs)
                    return (lhs + rhs) / 2
        
        df.loc[:,'mean_income'] = df['monthly_generated_income'].apply(calculate_mean)
        return df
    
    def age_transform(self, df:pd.DataFrame):
        df = df.copy()

        # Convert dates of birth to datetime objects
        df['client_date_of_birth'] = pd.to_datetime(df['client_date_of_birth'])

        # Calculate ages
        current_date = datetime.now()
        df['age'] = current_date.year - df['client_date_of_birth'].dt.year
        return df
    
    def payments(self, df_account:pd.DataFrame, df_payment:pd.DataFrame):
        df = df_payment.copy()
        df = df.groupby('account_qid',as_index=False).size().reset_index()
        df = df.rename(columns={'size': 'num_payment_term'})

        df_payment['year_month'] = pd.to_datetime(df_payment['effective_date']).dt.year.astype(str) + "-" \
            + pd.to_datetime(df_payment['effective_date']).dt.month.astype(str)

        df_Payments_freq_monthly = df_payment.groupby(['account_qid', 'year_month']).size().reset_index(name='freq')

        # Group by 'account_qid' and aggregate 'freq' into a list
        df_freq_list = df_Payments_freq_monthly.groupby('account_qid')['freq'].apply(list).reset_index()
        df['freq_payment_list'] = df_freq_list.freq
        df['max_payment_list'] = df_freq_list['freq'].apply(max)
        df['mean_payment_list'] = df_freq_list['freq'].apply(np.mean)

        # merge the account with payment
        df = df_account.merge(df, on=['account_qid'], how='left')
        return df
    
    def default(self, df:pd.DataFrame):
        # Define the mapping
        status_to_default = {
            "Disabled": 1,
            "Repossessed": 1,
            "Enabled": 0,
            "Unlocked": 0
        }

        # Apply the mapping
        df['default'] = df['status'].map(status_to_default)
        return df


    def days_active(self, df:pd.DataFrame):
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        #Define the target date
        target_date = pd.Timestamp('2024-04-02')

        #Calculate the ratio
        df['default_ratio'] = df['cumulative_days_disabled'] / (target_date - df['registration_date']).dt.days
        return df


    # Define other feature engineering functions as needed
    
    def transform(self, data, data_payment):
        # Main function to run through all feature engineering functions
        data = self.extract_PAYG(data)
        data = self.regist_time_extraction(data, 'registration_date') 
        data = self.regist_time_extraction(data, 'latest_payment_date')
        data = self.long_lat(data)
        data = self.mean_income(data)
        data = self.age_transform(data)
        data = self.payments(data,data_payment)
        data = self.default(data)
        data = self.days_active(data)

        # Call other feature engineering functions here
        data = data[self.numerical_feature + self.categorical_feature + [self.target['target1'],self.target['target2']]]

        return data
    
    
    
# Example usage
if __name__ == "__main__":
    # Sample data
    data1 = ...  # Load or generate your data
    data2 = ...  # Load or generate your data
    
    # Create an instance of FeatureEngineer
    FeatureEngineer = FeatureEngineer()
    
    # Apply feature engineering using the main function
    processed_data = FeatureEngineer.transform(data1, data2)
