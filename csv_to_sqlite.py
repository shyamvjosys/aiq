#!/usr/bin/env python3
"""
Convert CSV files to SQLite database with full-text search capabilities for NLP queries.
Preserves ALL columns from both CSV files without any data loss.
"""

import pandas as pd
import sqlite3
from sqlite_utils import Database
import os
import sys
import re

def clean_column_name(col_name):
    """Clean column names for SQLite compatibility while preserving readability."""
    # Replace problematic characters but keep the name readable
    cleaned = str(col_name).strip()
    # Replace spaces and special chars with underscores
    cleaned = re.sub(r'[^\w\-]', '_', cleaned)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = 'col_' + cleaned
    return cleaned if cleaned else 'unnamed_column'

def setup_database():
    """Create SQLite database from CSV files with FTS support."""
    
    # Read CSV files with all columns preserved
    print("Reading CSV files...")
    print("Loading josys-devices.csv...")
    devices_df = pd.read_csv('josys-devices.csv', dtype=str, na_filter=False)
    print(f"Devices CSV: {len(devices_df)} rows, {len(devices_df.columns)} columns")
    print("Device columns:", list(devices_df.columns))
    
    print("\nLoading josys-provisions.csv...")
    provisions_df = pd.read_csv('josys-provisions.csv', dtype=str, na_filter=False)
    print(f"Provisions CSV: {len(provisions_df)} rows, {len(provisions_df.columns)} columns")
    print("Provision columns:", list(provisions_df.columns))
    
    # Clean column names for SQLite compatibility
    print("\nCleaning column names...")
    original_device_cols = list(devices_df.columns)
    devices_df.columns = [clean_column_name(col) for col in devices_df.columns]
    
    original_provision_cols = list(provisions_df.columns)
    provisions_df.columns = [clean_column_name(col) for col in provisions_df.columns]
    
    # Print column mapping for verification
    print("\nDevice column mapping:")
    for orig, clean in zip(original_device_cols, devices_df.columns):
        if orig != clean:
            print(f"  '{orig}' -> '{clean}'")
    
    print("\nProvision column mapping:")
    for orig, clean in zip(original_provision_cols, provisions_df.columns):
        if orig != clean:
            print(f"  '{orig}' -> '{clean}'")
    
    print(f"\nFinal device columns ({len(devices_df.columns)}): {list(devices_df.columns)}")
    print(f"\nFinal provision columns ({len(provisions_df.columns)}): {list(provisions_df.columns)}")
    
    # Create database
    db_path = 'josys_data.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"\nCreating SQLite database: {db_path}")
    db = Database(db_path)
    
    # Insert data into tables - preserve ALL columns
    print("Creating devices table with ALL columns...")
    devices_records = devices_df.to_dict('records')
    db['devices'].insert_all(devices_records)
    print(f"  ✓ Inserted {len(devices_records)} device records with {len(devices_df.columns)} columns")
    
    print("Creating provisions table with ALL columns...")
    provisions_records = provisions_df.to_dict('records')
    db['provisions'].insert_all(provisions_records)
    print(f"  ✓ Inserted {len(provisions_records)} provision records with {len(provisions_df.columns)} columns")
    
    # Enable FTS (Full-Text Search) on key searchable columns
    print("\nSetting up full-text search...")
    
    # FTS for devices table - identify searchable text columns
    device_text_columns = []
    for col in devices_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in [
            'asset', 'device', 'type', 'manufacturer', 'model', 'system', 
            'information', 'email', 'city', 'region', 'status', 'vendor'
        ]):
            device_text_columns.append(col)
    
    print(f"Device FTS columns: {device_text_columns}")
    if device_text_columns:
        try:
            db['devices'].enable_fts(device_text_columns, create_triggers=True)
            print(f"  ✓ Enabled FTS on {len(device_text_columns)} device columns")
        except Exception as e:
            print(f"  ⚠ FTS setup warning for devices: {e}")
    
    # FTS for provisions table - identify searchable text columns
    provision_text_columns = []
    for col in provisions_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in [
            'first', 'last', 'name', 'user', 'email', 'username', 
            'role', 'status', 'location'
        ]):
            provision_text_columns.append(col)
    
    print(f"Provision FTS columns: {provision_text_columns}")
    if provision_text_columns:
        try:
            db['provisions'].enable_fts(provision_text_columns, create_triggers=True)
            print(f"  ✓ Enabled FTS on {len(provision_text_columns)} provision columns")
        except Exception as e:
            print(f"  ⚠ FTS setup warning for provisions: {e}")
    
    # Create helpful views using the actual column names
    print("\nCreating helpful views...")
    
    # Find the actual column names in the cleaned format
    asset_num_col = next((col for col in devices_df.columns if 'asset' in col.lower() and 'number' in col.lower()), 'Asset_Number')
    device_type_col = next((col for col in devices_df.columns if 'device' in col.lower() and 'type' in col.lower()), 'Device_Type')
    device_status_col = next((col for col in devices_df.columns if 'device' in col.lower() and 'status' in col.lower()), 'Device_Status')
    manufacturer_col = next((col for col in devices_df.columns if 'manufacturer' in col.lower()), 'Manufacturer')
    model_name_col = next((col for col in devices_df.columns if 'model' in col.lower() and 'name' in col.lower()), 'Model_Name')
    assigned_email_col = next((col for col in devices_df.columns if 'assigned' in col.lower() and 'email' in col.lower()), 'Assigned_User_s_Email')
    city_col = next((col for col in devices_df.columns if 'city' in col.lower()), 'City')
    region_col = next((col for col in devices_df.columns if 'region' in col.lower()), 'Region')
    additional_info_col = next((col for col in devices_df.columns if 'additional' in col.lower() and 'information' in col.lower()), 'Additional_Information')
    
    user_id_col = next((col for col in provisions_df.columns if 'user' in col.lower() and 'id' in col.lower()), 'User_ID')
    first_name_col = next((col for col in provisions_df.columns if 'first' in col.lower() and 'name' in col.lower()), 'First_Name')
    last_name_col = next((col for col in provisions_df.columns if 'last' in col.lower() and 'name' in col.lower()), 'Last_Name')
    email_col = next((col for col in provisions_df.columns if col.lower() == 'email'), 'Email')
    status_col = next((col for col in provisions_df.columns if col.lower() == 'status'), 'Status')
    role_col = next((col for col in provisions_df.columns if col.lower() == 'role'), 'Role')
    work_location_col = next((col for col in provisions_df.columns if 'work' in col.lower() and 'location' in col.lower()), 'Work_Location_Code')
    
    # Create views with dynamic column names
    active_devices_view = f'''
        CREATE VIEW IF NOT EXISTS active_devices_with_users AS
        SELECT 
            {asset_num_col} as Asset_Number,
            {device_type_col} as Device_Type,
            {manufacturer_col} as Manufacturer,
            {model_name_col} as Model_Name,
            {device_status_col} as Device_Status,
            {assigned_email_col} as Assigned_Users_Email,
            {city_col} as City,
            {region_col} as Region,
            {additional_info_col} as Additional_Information
        FROM devices 
        WHERE {device_status_col} = 'In-use' AND {assigned_email_col} IS NOT NULL AND {assigned_email_col} != '';
    '''
    
    user_summary_view = f'''
        CREATE VIEW IF NOT EXISTS user_access_summary AS
        SELECT 
            {user_id_col} as User_ID,
            {first_name_col} as First_Name,
            {last_name_col} as Last_Name,
            {email_col} as Email,
            {status_col} as Status,
            {role_col} as Role,
            {work_location_col} as Work_Location_Code
        FROM provisions 
        WHERE {status_col} = 'Active';
    '''
    
    try:
        db.executescript(active_devices_view)
        print("  ✓ Created active_devices_with_users view")
    except Exception as e:
        print(f"  ⚠ View creation warning: {e}")
    
    try:
        db.executescript(user_summary_view)
        print("  ✓ Created user_access_summary view")
    except Exception as e:
        print(f"  ⚠ View creation warning: {e}")
    
    # Verify the database
    print(f"\n✅ Database created successfully: {db_path}")
    print(f"   📊 Devices table: {len(devices_df)} records, {len(devices_df.columns)} columns")
    print(f"   👥 Provisions table: {len(provisions_df)} records, {len(provisions_df.columns)} columns")
    
    # Show table schemas
    print(f"\n📋 Table schemas:")
    print(f"   Devices columns: {', '.join(devices_df.columns)}")
    print(f"   Provisions columns: {', '.join(provisions_df.columns[:10])}... (showing first 10 of {len(provisions_df.columns)})")
    
    return db_path

if __name__ == "__main__":
    setup_database()
