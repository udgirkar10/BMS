"""
Extract real-time battery data from PostgreSQL database and save to CSV
Maps database columns to model feature names
"""

import psycopg2
import pandas as pd
from datetime import datetime
import time
import os


class BatteryDataExtractor:
    """
    Extracts battery data from PostgreSQL and formats it for the RUL model
    """
    
    def __init__(self, db_config):
        """
        Initialize database connection
        
        Args:
            db_config: Dictionary with database connection parameters
                {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'battery_db',
                    'user': 'your_username',
                    'password': 'your_password'
                }
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # Define mapping from database columns to model features
        # Updated to match your BatteryLog model schema
        self.column_mapping = {
            # Database column name : Model feature name
            'timestamp': 'timestamp',
            'present_current': 'Battery_Current',
            'present_voltage': 'Battery_Voltage',
            'temp': 'Battery_Temp',
            'soh': 'Battery_SoH',
            'soe': 'Estimated_SoE',
            'soc': 'Estimated_Soc',
            'estimated_capacity': 'Estimated_Battery_Capacity',
            'estimated_range': 'estimated_range',
            'led_overcurrent': 'LED_OverCurrent',
            'led_undertemp': 'LED_UnderTemp',
            'led_overtemp': 'LED_OverTemp',
            'led_undervoltage': 'LED_UnderVoltage',
            'led_overvoltage': 'LED_OverVoltage',
            'pack_current': 'Pack_Current',
            'pack_voltage': 'Pack_Voltage',
            'sop': 'SoP',
            'charge_flag': 'Charging_Status',
            'charge_discharge_cycle': 'Charge_Discharge_Cycles',
            'charging_time': 'Time_To_Charge'
        }
        
        # Features NOT in your database (will be filled with defaults)
        self.missing_features = {
            'Vehicle_speed': 0.0,           # Not in your DB
            'Distance_Travelled': 0.0       # Not in your DB
        }
        
        # Required model features (22 total)
        self.required_features = [
            'timestamp', 'Battery_Current', 'Battery_Voltage', 'Battery_Temp',
            'Battery_SoH', 'Estimated_SoE', 'Estimated_Soc', 
            'Estimated_Battery_Capacity', 'estimated_range', 'Vehicle_speed',
            'Distance_Travelled', 'LED_OverCurrent', 'LED_UnderTemp',
            'LED_OverTemp', 'LED_UnderVoltage', 'LED_OverVoltage',
            'Pack_Current', 'Pack_Voltage', 'SoP', 'Charging_Status',
            'Charge_Discharge_Cycles', 'Time_To_Charge'
        ]
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("✓ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Disconnected from database")
    
    def extract_data(self, table_name, vin=None, start_time=None, end_time=None, limit=None):
        """
        Extract data from PostgreSQL table
        
        Args:
            table_name: Name of the table containing battery data (e.g., 'battery_log')
            vin: Vehicle VIN to filter by (optional)
            start_time: Start timestamp (optional, format: 'YYYY-MM-DD HH:MM:SS')
            end_time: End timestamp (optional, format: 'YYYY-MM-DD HH:MM:SS')
            limit: Maximum number of rows to extract (optional)
            
        Returns:
            pandas DataFrame with extracted data
        """
        try:
            # Build SQL query with columns from your BatteryLog model
            db_columns = list(self.column_mapping.keys())
            columns_str = ', '.join(db_columns)
            
            query = f"SELECT {columns_str} FROM {table_name}"
            
            # Add filters if provided
            conditions = []
            if vin:
                conditions.append(f"vin = '{vin}'")
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Order by timestamp (chronological)
            query += " ORDER BY timestamp ASC"
            
            # Add limit if provided
            if limit:
                query += f" LIMIT {limit}"
            
            print(f"Executing query: {query}")
            
            # Execute query
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=db_columns)
            
            print(f"✓ Extracted {len(df)} rows from database")
            
            return df
            
        except Exception as e:
            print(f"✗ Data extraction failed: {e}")
            return None
    
    def map_columns(self, df):
        """
        Map database columns to model feature names
        
        Args:
            df: DataFrame with database column names
            
        Returns:
            DataFrame with model feature names
        """
        # Rename columns according to mapping
        df_mapped = df.rename(columns=self.column_mapping)
        
        # Add missing features with default values
        for feature, default_value in self.missing_features.items():
            df_mapped[feature] = default_value
            print(f"  Added missing feature '{feature}' with default value: {default_value}")
        
        # Check if all required features are present
        missing_features = set(self.required_features) - set(df_mapped.columns)
        if missing_features:
            print(f"⚠ Warning: Still missing features: {missing_features}")
            print("  These features will be filled with default values")
            
            # Fill any remaining missing features with default values
            for feature in missing_features:
                if feature == 'timestamp':
                    df_mapped[feature] = pd.Timestamp.now()
                elif 'LED_' in feature:
                    df_mapped[feature] = False
                else:
                    df_mapped[feature] = 0.0
        
        # Convert string columns to appropriate types
        # Your DB stores many values as strings, need to convert them
        numeric_features = [
            'Battery_Current', 'Battery_Voltage', 'Battery_Temp', 'Battery_SoH',
            'Estimated_SoE', 'Estimated_Soc', 'Estimated_Battery_Capacity',
            'estimated_range', 'Time_To_Charge'
        ]
        
        for feature in numeric_features:
            if feature in df_mapped.columns:
                df_mapped[feature] = pd.to_numeric(df_mapped[feature], errors='coerce').fillna(0.0)
        
        # Convert boolean columns
        boolean_features = [
            'LED_OverCurrent', 'LED_UnderTemp', 'LED_OverTemp',
            'LED_UnderVoltage', 'LED_OverVoltage', 'Charging_Status'
        ]
        
        for feature in boolean_features:
            if feature in df_mapped.columns:
                df_mapped[feature] = df_mapped[feature].astype(bool)
        
        # Reorder columns to match required order
        df_mapped = df_mapped[self.required_features]
        
        print(f"✓ Mapped columns to model features")
        
        return df_mapped
    
    def save_to_csv(self, df, output_file='battery_data.csv', append=False):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            output_file: Output CSV file path
            append: If True, append to existing file; if False, overwrite
        """
        try:
            mode = 'a' if append else 'w'
            header = not append or not os.path.exists(output_file)
            
            df.to_csv(output_file, mode=mode, header=header, index=False)
            
            action = "Appended to" if append else "Saved to"
            print(f"✓ {action} {output_file} ({len(df)} rows)")
            
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")
    
    def extract_and_save(self, table_name, output_file='battery_data.csv', 
                        vin=None, start_time=None, end_time=None, limit=None, append=False):
        """
        Complete workflow: Extract, map, and save data
        
        Args:
            table_name: Database table name (e.g., 'battery_log')
            output_file: Output CSV file path
            vin: Vehicle VIN to filter by (optional)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum rows (optional)
            append: Append to existing file (optional)
        """
        if not self.connect():
            return False
        
        try:
            # Extract data
            df = self.extract_data(table_name, vin, start_time, end_time, limit)
            
            if df is None or len(df) == 0:
                print("✗ No data extracted")
                return False
            
            # Map columns
            df_mapped = self.map_columns(df)
            
            # Save to CSV
            self.save_to_csv(df_mapped, output_file, append)
            
            print(f"\n✓ Successfully extracted {len(df_mapped)} rows")
            print(f"  Output file: {output_file}")
            
            return True
            
        finally:
            self.disconnect()
    
    def continuous_extraction(self, table_name, output_file='battery_data.csv', 
                             vin=None, interval_seconds=60, batch_size=100):
        """
        Continuously extract new data at regular intervals
        
        Args:
            table_name: Database table name (e.g., 'battery_log')
            output_file: Output CSV file path
            vin: Vehicle VIN to filter by (optional)
            interval_seconds: Time between extractions (seconds)
            batch_size: Number of rows to extract per batch
        """
        print(f"Starting continuous extraction (interval: {interval_seconds}s)")
        if vin:
            print(f"Filtering by VIN: {vin}")
        print("Press Ctrl+C to stop\n")
        
        last_timestamp = None
        
        try:
            while True:
                if not self.connect():
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
                
                try:
                    # Extract only new data since last timestamp
                    df = self.extract_data(
                        table_name,
                        vin=vin,
                        start_time=last_timestamp,
                        limit=batch_size
                    )
                    
                    if df is not None and len(df) > 0:
                        # Map and save
                        df_mapped = self.map_columns(df)
                        self.save_to_csv(df_mapped, output_file, append=True)
                        
                        # Update last timestamp
                        last_timestamp = df['timestamp'].max()
                        
                        print(f"[{datetime.now()}] Extracted {len(df)} new rows")
                    else:
                        print(f"[{datetime.now()}] No new data")
                    
                except Exception as e:
                    print(f"✗ Error during extraction: {e}")
                
                finally:
                    self.disconnect()
                
                # Wait before next extraction
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n✓ Continuous extraction stopped")


# Example usage
if __name__ == "__main__":
    # Database configuration
    # MODIFY THESE VALUES TO MATCH YOUR DATABASE
    db_config = {
        'host': 'localhost',        # Database host
        'port': 5432,                # Database port
        'database': 'battery_db',    # Database name
        'user': 'your_username',     # Your username
        'password': 'your_password'  # Your password
    }
    
    # Table name from your BatteryLog model
    table_name = 'battery_log'  # Your table name (likely 'battery_log')
    
    # Optional: Filter by specific vehicle VIN
    vehicle_vin = None  # Set to specific VIN like 'VIN123456' or None for all vehicles
    
    # Initialize extractor
    extractor = BatteryDataExtractor(db_config)
    
    # Option 1: One-time extraction
    print("=== One-time Extraction ===")
    extractor.extract_and_save(
        table_name=table_name,
        output_file='battery_data.csv',
        vin=vehicle_vin,                    # Optional: filter by VIN
        # start_time='2025-01-01 00:00:00',  # Optional: start time
        # end_time='2025-12-31 23:59:59',    # Optional: end time
        # limit=10000,                        # Optional: max rows
        append=False  # Set to True to append to existing file
    )
    
    # Option 2: Continuous extraction (uncomment to use)
    # print("\n=== Continuous Extraction ===")
    # extractor.continuous_extraction(
    #     table_name=table_name,
    #     output_file='battery_data.csv',
    #     vin=vehicle_vin,          # Optional: filter by VIN
    #     interval_seconds=60,      # Extract every 60 seconds
    #     batch_size=100            # Extract 100 rows per batch
    # )
