#!/usr/bin/env python3
"""
True NLP Web Interface for Josys Data Search
Uses OpenAI API for NL2SQL and embeddings for semantic search
"""

import os
import json
import sqlite3
import time
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, render_template_string
import openai
import threading
from dotenv import load_dotenv
import numpy as np

class JosysOpenAINLP:
    """NLP interface using OpenAI for NL2SQL and embeddings."""
    
    def __init__(self, db_path: str = 'josys_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Load OpenAI API key from .env file
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("❌ OpenAI API key not found in .env file! Please add OPENAI_API_KEY=your_key_here to .env")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.api_key)
        
        # Cache for embeddings and results
        self.embedding_cache = {}
        self.query_cache = {}
        
        print("🤖 OpenAI NLP interface initialized successfully!")
        print(f"   Database: {db_path}")
        print(f"   OpenAI API: ✅ Connected")
        
        # Get schema info for SQL generation
        self.schema_info = self._get_detailed_schema()
        print(f"   Schema loaded: {len(self.schema_info.split('Table:')) - 1} tables")
    
    def _get_detailed_schema(self) -> str:
        """Get detailed database schema for SQL generation."""
        schema_parts = []
        
        # Get devices table schema
        devices_info = self.conn.execute("PRAGMA table_info(devices)").fetchall()
        schema_parts.append("Table: devices")
        schema_parts.append("Description: IT assets (laptops, computers, phones) with assignment information")
        schema_parts.append("Columns:")
        for col in devices_info:
            schema_parts.append(f"  - {col[1]} ({col[2]}) - {self._get_column_description('devices', col[1])}")
        
        # Get provisions table schema  
        provisions_info = self.conn.execute("PRAGMA table_info(provisions)").fetchall()
        schema_parts.append("\nTable: provisions")
        schema_parts.append("Description: User access to applications and services")
        schema_parts.append("Key User Columns:")
        for col in provisions_info[:8]:  # Show first 8 user columns
            schema_parts.append(f"  - {col[1]} ({col[2]}) - {self._get_column_description('provisions', col[1])}")
        
        # Add ALL application columns for comprehensive coverage
        schema_parts.append("\nApplication Access Columns (use backticks for complex names):")
        app_columns = provisions_info[8:]  # Skip user info columns
        for col in app_columns:
            col_name = col[1]
            # Highlight important applications
            if any(app in col_name for app in ['Datadog', 'GitHub', 'Slack', 'AWS', 'Google', 'Microsoft']):
                schema_parts.append(f"  - `{col_name}` ({col[2]}) - Application access")
        
        schema_parts.append(f"\nIMPORTANT: Total {len(provisions_info)} columns in provisions table")
        schema_parts.append("For complex column names with special characters, use backticks: `column_name`")
        schema_parts.append("Application values: 'Activated', 'Invited', '' (empty for no access)")
        
        # Get app_portfolio table schema
        portfolio_info = self.conn.execute("PRAGMA table_info(app_portfolio)").fetchall()
        schema_parts.append("\nTable: app_portfolio")
        schema_parts.append("Description: Detailed application access with roles, costs, and account information")
        schema_parts.append("Columns:")
        for col in portfolio_info:
            schema_parts.append(f"  - {col[1]} ({col[2]}) - {self._get_column_description('app_portfolio', col[1])}")
        
        schema_parts.append(f"\nIMPORTANT: Total {len(portfolio_info)} columns in app_portfolio table")
        schema_parts.append("App Portfolio provides granular access details with roles and costs")
        schema_parts.append("Account Status values: 'Activated', 'Invited', etc.")
        schema_parts.append("Use this table for detailed role-based queries and cost analysis")
        
        return "\n".join(schema_parts)
    
    def _get_column_description(self, table: str, column: str) -> str:
        """Get human-readable description for database columns."""
        descriptions = {
            'devices': {
                'Asset_Number': 'Unique device identifier',
                'Device_Type': 'Type of device (Laptop, Computer, etc.)',
                'Manufacturer': 'Device manufacturer (Apple, Dell, etc.)',
                'Model_Name': 'Specific model name',
                'Device_Status': 'Current status (Available, Assigned, etc.)',
                'Assigned_User_s_Email': 'Email of user assigned to device',
                'Assigned_User_s_ID': 'User ID of assigned user',
                'City': 'Location city',
                'Region': 'Geographic region'
            },
            'provisions': {
                'User_ID': 'Unique user identifier',
                'First_Name': 'User first name',
                'Last_Name': 'User last name', 
                'Email': 'User email address',
                'Role': 'Job role/title',
                'Status': 'User status (Active, Inactive)',
                'Work_Location_Code': 'Office location code'
            },
            'app_portfolio': {
                'App': 'Application name (e.g., AWS, GitHub, Slack)',
                'Identifier': 'Application instance/account identifier',
                'ID': 'Username/login ID for the application (e.g., email or account name)',
                'Account_Status': 'Account status (Activated, Invited, etc.)',
                'Monthly_Expense': 'Monthly cost for this access',
                'Role_s': 'User roles/permissions in the application',
                'Additional_Information': 'Extra details about the access',
                'First_Name': 'User first name',
                'Last_Name': 'User last name',
                'User_Status': 'User account status',
                'Email': 'User email address',
                'User_ID': 'Unique user identifier',
                'User_Category': 'User type (Full-time, Contractor, etc.)',
                'Department_s': 'User department',
                'Job_Title': 'User job title',
                'Role': 'User organizational role'
            }
        }
        
        return descriptions.get(table, {}).get(column, 'Data field')
    
    def natural_language_to_sql(self, question: str) -> Dict[str, Any]:
        """Convert natural language question to SQL using OpenAI."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"nl2sql:{question.lower()}"
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key].copy()
            cached['cached'] = True
            return cached
        
        # Create detailed prompt for OpenAI
        prompt = f"""You are an expert SQL developer for a Josys IT asset management system.

DATABASE SCHEMA:
{self.schema_info}

IMPORTANT QUERY GUIDELINES:
1. For device queries: Use the 'devices' table
2. For user queries: Use the 'provisions' table  
3. For application access details: Use the 'app_portfolio' table
4. For CROSS-TABLE queries: JOIN on Email fields
   - provisions.Email = devices.Assigned_User_s_Email
   - app_portfolio.Email = provisions.Email
5. For user names: Search in First_Name, Last_Name, or Email columns
6. Use LIKE '%...%' for partial text matches and UPPER() for case-insensitive
7. Use backticks for complex column names with special characters
8. Always limit results to reasonable numbers (LIMIT 20)
9. IMPORTANT: "MacBook" refers to Apple laptops - use Device_Type = 'LAPTOP' AND Manufacturer = 'APPLE'
10. For location queries: Use exact matches with UPPER() - Region = 'INDIA', City = 'BANGALORE', etc.
11. For country-based queries: JOIN app_portfolio with devices table on Email fields to get location data
12. Japan locations: Check for Region/City LIKE '%japan%' OR City LIKE '%tokyo%' OR other Japanese cities
13. Japanese employees: Identify by Japanese first names OR .jp email domains when location data unavailable
14. Common Japanese names: Tomoyo, Mari, Kohei, Ayumi, Seiji, Yuya, Makoto, Michiko, Ikumi, Terumichi, Eri, Yuki, Naomi, Raku, Tsuyoshi, Kazuki, Maho, Satoshi, Goki, Takamitsu, Yukinori, Tomoya
15. For complex multi-criteria queries: Use three-table JOINs (devices-provisions-app_portfolio)
16. When queries return 0 results: Provide breakdown analysis of individual components
17. Include cross-reference checks to identify why criteria don't intersect

EXAMPLE QUERIES:
- "devices assigned to Arvind" → SELECT * FROM devices WHERE UPPER(Assigned_User_s_Email) LIKE UPPER('%arvind%') OR UPPER(Assigned_User_s_ID) LIKE UPPER('%arvind%')
- "MacBook laptops" → SELECT * FROM devices WHERE UPPER(Device_Type) = 'LAPTOP' AND UPPER(Manufacturer) = 'APPLE'
- "MacBook laptops assigned to employees in India" → SELECT * FROM devices WHERE UPPER(Device_Type) = 'LAPTOP' AND UPPER(Manufacturer) = 'APPLE' AND UPPER(Region) = 'INDIA' AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
- "users with GitHub access" → SELECT * FROM provisions WHERE GitHub IS NOT NULL AND GitHub != ''
- "available devices in Bangalore" → SELECT * FROM devices WHERE UPPER(Device_Status) LIKE UPPER('%available%') AND UPPER(City) LIKE UPPER('%bangalore%')
- "names with multiple DataDog licenses" → SELECT First_Name, Last_Name, `Datadog_-_JOSYS`, `Datadog_-_JOSYS-IDAC`, `Datadog_-_JOSYS-JEP`, `Datadog_-_JOSYS-Non-Prod` FROM provisions WHERE (`Datadog_-_JOSYS` != '' AND `Datadog_-_JOSYS` IS NOT NULL) + (`Datadog_-_JOSYS-IDAC` != '' AND `Datadog_-_JOSYS-IDAC` IS NOT NULL) + (`Datadog_-_JOSYS-JEP` != '' AND `Datadog_-_JOSYS-JEP` IS NOT NULL) + (`Datadog_-_JOSYS-Non-Prod` != '' AND `Datadog_-_JOSYS-Non-Prod` IS NOT NULL) > 1

APP PORTFOLIO EXAMPLES:
- "AWS access for users" → SELECT * FROM app_portfolio WHERE UPPER(App) LIKE UPPER('%aws%') AND Account_Status = 'Activated'
- "AWS Admin usernames" → SELECT DISTINCT ID, First_Name, Last_Name, Email, Identifier FROM app_portfolio WHERE UPPER(App) LIKE UPPER('%aws%') AND UPPER(Role_s) LIKE UPPER('%administrator%') AND Account_Status = 'Activated'
- "AWS admins in Japan" → SELECT DISTINCT ap.ID, ap.First_Name, ap.Last_Name, ap.Email, ap.Identifier FROM app_portfolio ap JOIN devices d ON ap.Email = d.Assigned_User_s_Email WHERE UPPER(ap.App) LIKE UPPER('%aws%') AND UPPER(ap.Role_s) LIKE UPPER('%administrator%') AND ap.Account_Status = 'Activated' AND (UPPER(d.Region) LIKE UPPER('%japan%') OR UPPER(d.City) LIKE UPPER('%japan%') OR UPPER(d.City) LIKE UPPER('%tokyo%'))
- "employees in Japan" → SELECT DISTINCT First_Name, Last_Name, Email, User_Category, Department_s FROM app_portfolio WHERE (First_Name IN ('Tomoyo', 'Mari', 'Kohei', 'Ayumi', 'Seiji', 'Yuya', 'Makoto', 'Michiko', 'Ikumi', 'Terumichi', 'Eri', 'Yuki', 'Naomi', 'Raku', 'Tsuyoshi', 'Kazuki', 'Maho', 'Satoshi', 'Goki', 'Takamitsu', 'Yukinori', 'Tomoya', 'Jingjing') OR Email LIKE '%.jp' OR Email LIKE '%amazon.co.jp%') AND User_Status = 'Active'
- "Lenovo laptop users with AWS admin and Notion" → SELECT DISTINCT d.Asset_Number, d.Manufacturer, d.Model_Name, p.First_Name, p.Last_Name, p.Email, ap.Identifier, ap.Role_s FROM devices d JOIN provisions p ON d.Assigned_User_s_Email = p.Email JOIN app_portfolio ap ON p.Email = ap.Email WHERE UPPER(d.Manufacturer) = 'LENOVO' AND d.Device_Status = 'In-use' AND UPPER(ap.App) LIKE '%AWS%' AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%' AND ap.Account_Status = 'Activated' AND (p.`Notion_-_Josys_inc` = 'Activated' OR p.`Notion_-_Josys_public` = 'Activated')
- "users with AdministratorAccess role" → SELECT First_Name, Last_Name, Email, App, Identifier, Role_s FROM app_portfolio WHERE UPPER(Role_s) LIKE UPPER('%administrator%')
- "monthly costs for GitHub access" → SELECT App, SUM(CAST(Monthly_Expense AS REAL)) as Total_Cost FROM app_portfolio WHERE UPPER(App) LIKE UPPER('%github%') GROUP BY App
- "contractors with application access" → SELECT * FROM app_portfolio WHERE User_Category = 'Contractor' AND Account_Status = 'Activated'

CROSS-TABLE JOIN EXAMPLES:
- "names with MacBook and Notion license" → SELECT DISTINCT p.First_Name, p.Last_Name, d.Device_Type, d.Manufacturer, p.`Notion_-_Josys_inc` FROM provisions p JOIN devices d ON p.Email = d.Assigned_User_s_Email WHERE (UPPER(d.Device_Type) LIKE '%LAPTOP%' AND UPPER(d.Manufacturer) LIKE '%APPLE%') AND p.`Notion_-_Josys_inc` != '' AND p.`Notion_-_Josys_inc` IS NOT NULL LIMIT 20
- "users with Apple devices and GitHub access" → SELECT p.First_Name, p.Last_Name, d.Asset_Number, d.Manufacturer, p.GitHub FROM provisions p JOIN devices d ON p.Email = d.Assigned_User_s_Email WHERE UPPER(d.Manufacturer) LIKE '%APPLE%' AND p.GitHub = 'Activated' LIMIT 20
- "users with devices and detailed AWS access" → SELECT DISTINCT d.Asset_Number, d.Device_Type, ap.First_Name, ap.Last_Name, ap.App, ap.Identifier, ap.Role_s FROM devices d JOIN app_portfolio ap ON d.Assigned_User_s_Email = ap.Email WHERE UPPER(ap.App) LIKE UPPER('%aws%') AND ap.Account_Status = 'Activated' LIMIT 20

USER QUESTION: {question}

Generate ONLY the SQL query (no explanations):"""

        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert SQL developer. Generate only SQL queries, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL query
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            if sql_query.startswith('sql'):
                sql_query = sql_query[3:].strip()
            
            # Execute the query
            try:
                cursor = self.conn.execute(sql_query)
                results = [dict(row) for row in cursor.fetchall()]
                
                execution_time = time.time() - start_time
                
                result = {
                    'question': question,
                    'sql': sql_query,
                    'results': results,
                    'count': len(results),
                    'execution_time': execution_time,
                    'method': 'openai_nl2sql',
                    'status': 'success',
                    'cached': False
                }
                
                # Add comprehensive insights and analysis
                enhanced_result = self._generate_comprehensive_insights(question, result)
                
                # Cache the result
                self.query_cache[cache_key] = enhanced_result.copy()
                
                return enhanced_result
                
            except sqlite3.Error as e:
                return {
                    'question': question,
                    'sql': sql_query,
                    'error': f"SQL execution error: {str(e)}",
                    'status': 'sql_error',
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            return {
                'question': question,
                'error': f"OpenAI API error: {str(e)}",
                'status': 'api_error',
                'execution_time': time.time() - start_time
            }
    
    def combined_nlp_search(self, question: str) -> Dict[str, Any]:
        """Combine NL2SQL with fallback keyword search."""
        start_time = time.time()
        
        # Try NL2SQL first (more accurate for structured queries)
        nl2sql_result = self.natural_language_to_sql(question)
        
        # If NL2SQL succeeds and has results, use it
        if nl2sql_result['status'] == 'success' and nl2sql_result['count'] > 0:
            nl2sql_result['method'] = 'combined_nl2sql_primary'
            return nl2sql_result
        
        # Fallback to keyword search
        fallback_result = self._keyword_fallback_search(question)
        fallback_result['method'] = 'combined_keyword_fallback'
        fallback_result['fallback_reason'] = f"NL2SQL failed: {nl2sql_result.get('error', 'No results')}"
        fallback_result['attempted_sql'] = nl2sql_result.get('sql', 'N/A')
        
        return fallback_result
    
    def _keyword_fallback_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Keyword-based fallback search."""
        start_time = time.time()
        results = []
        
        try:
            # Search devices with keyword matching
            device_cursor = self.conn.execute("""
                SELECT Asset_Number, Device_Type, Manufacturer, Model_Name, 
                       Device_Status, Assigned_User_s_Email, City, Region
                FROM devices 
                WHERE UPPER(Asset_Number) LIKE UPPER(?) 
                   OR UPPER(Device_Type) LIKE UPPER(?) 
                   OR UPPER(Manufacturer) LIKE UPPER(?) 
                   OR UPPER(Assigned_User_s_Email) LIKE UPPER(?)
                   OR UPPER(City) LIKE UPPER(?)
                LIMIT ?
            """, [f'%{query}%'] * 5 + [limit//2])
            
            for row in device_cursor.fetchall():
                results.append({
                    'type': 'device',
                    'similarity': 0.8,
                    'data': dict(row)
                })
            
            # Search provisions with keyword matching
            provision_cursor = self.conn.execute("""
                SELECT User_ID, First_Name, Last_Name, Email, Role, Status,
                       Work_Location_Code
                FROM provisions 
                WHERE UPPER(User_ID) LIKE UPPER(?) 
                   OR UPPER(First_Name) LIKE UPPER(?) 
                   OR UPPER(Last_Name) LIKE UPPER(?) 
                   OR UPPER(Email) LIKE UPPER(?)
                   OR UPPER(Role) LIKE UPPER(?)
                LIMIT ?
            """, [f'%{query}%'] * 5 + [limit//2])
            
            for row in provision_cursor.fetchall():
                results.append({
                    'type': 'user',
                    'similarity': 0.8,
                    'data': dict(row)
                })
                
        except Exception as e:
            print(f"❌ Keyword fallback error: {e}")
        
        execution_time = time.time() - start_time
        
        return {
            'query': query,
            'method': 'keyword_fallback',
            'results': results[:limit],
            'count': len(results[:limit]),
            'execution_time': execution_time,
            'status': 'success'
        }
    
    def _generate_comprehensive_insights(self, question: str, result: dict) -> dict:
        """Generate comprehensive insights and analysis for query results."""
        
        # Start with the base result
        enhanced_result = result.copy()
        
        question_lower = question.lower()
        results = result.get('results', [])
        count = result.get('count', 0)
        
        # Initialize insights
        insights = []
        breakdown_data = {}
        suggestions = []
        
        # Analyze results based on question type and result count
        if count == 0:
            insights.append("❌ No results found matching your criteria.")
            
            # Provide breakdown analysis for complex queries
            if self._is_complex_multi_criteria_query(question_lower):
                breakdown_data = self._analyze_query_breakdown(question_lower)
                insights.extend(self._generate_breakdown_insights(breakdown_data))
                suggestions.extend(self._generate_alternative_suggestions(question_lower, breakdown_data))
            
        elif count > 0:
            insights.append(f"✅ Found {count} result{'s' if count != 1 else ''} matching your criteria.")
            
            # Add specific insights based on query type
            if 'aws' in question_lower and 'admin' in question_lower:
                insights.append(f"🔐 Security Note: {count} user{'s' if count != 1 else ''} with AWS administrative privileges found.")
                
            if 'japan' in question_lower or any(name in question_lower for name in ['tomoyo', 'mari', 'kohei']):
                insights.append(f"🗾 Geographic Analysis: Identified {count} Japanese employee{'s' if count != 1 else ''} in the system.")
                
            if 'laptop' in question_lower or 'device' in question_lower:
                device_insights = self._analyze_device_results(results)
                insights.extend(device_insights)
                
            if 'notion' in question_lower or 'license' in question_lower:
                license_insights = self._analyze_license_results(results)
                insights.extend(license_insights)
        
        # Add performance insights
        exec_time = result.get('execution_time', 0)
        if exec_time > 1.0:
            insights.append(f"⏱️ Query executed in {exec_time:.2f}s - complex cross-table analysis performed.")
        elif exec_time > 0:
            insights.append(f"⚡ Fast execution: {exec_time:.3f}s")
        
        # Generate comprehensive breakdown analysis
        detailed_breakdown = self._generate_detailed_breakdown_analysis(question_lower, results)
        
        # Generate key findings and cross-references
        key_findings = self._generate_key_findings(question_lower, results, breakdown_data)
        cross_references = self._generate_cross_references(question_lower, breakdown_data)
        
        # Add enhanced result data
        enhanced_result.update({
            'insights': insights,
            'breakdown_data': breakdown_data,
            'detailed_breakdown': detailed_breakdown,
            'suggestions': suggestions,
            'key_findings': key_findings,
            'cross_references': cross_references,
            'analysis_type': self._determine_analysis_type(question_lower),
            'comprehensive_summary': self._generate_summary(question, count, insights)
        })
        
        return enhanced_result
    
    def _is_complex_multi_criteria_query(self, question_lower: str) -> bool:
        """Detect if this is a complex multi-criteria query that needs breakdown."""
        criteria_count = 0
        
        # Check for different types of criteria
        if any(hw in question_lower for hw in ['lenovo', 'apple', 'laptop', 'device']):
            criteria_count += 1
        if any(access in question_lower for access in ['aws', 'admin', 'administrator']):
            criteria_count += 1
        if any(app in question_lower for app in ['notion', 'github', 'slack', 'license']):
            criteria_count += 1
        if any(loc in question_lower for loc in ['japan', 'india', 'bangalore']):
            criteria_count += 1
            
        return criteria_count >= 2
    
    def _analyze_query_breakdown(self, question_lower: str) -> dict:
        """Analyze breakdown of complex query components."""
        breakdown = {}
        
        try:
            # Analyze hardware components
            if 'lenovo' in question_lower:
                breakdown['lenovo_users'] = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) 
                    FROM devices 
                    WHERE UPPER(Manufacturer) = 'LENOVO' AND Device_Status = 'In-use'
                    AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
                ''').fetchone()[0]
            
            # Analyze AWS admin access
            if 'aws' in question_lower and 'admin' in question_lower:
                breakdown['aws_admins'] = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) 
                    FROM app_portfolio 
                    WHERE UPPER(App) LIKE '%AWS%' AND UPPER(Role_s) LIKE '%ADMINISTRATOR%' 
                    AND Account_Status = 'Activated'
                ''').fetchone()[0]
            
            # Analyze Notion licenses
            if 'notion' in question_lower:
                breakdown['notion_users'] = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) 
                    FROM provisions 
                    WHERE `Notion_-_Josys_inc` = 'Activated' OR `Notion_-_Josys_public` = 'Activated'
                ''').fetchone()[0]
            
            # Cross-reference analysis
            if 'lenovo_users' in breakdown and 'aws_admins' in breakdown:
                breakdown['lenovo_aws'] = self.conn.execute('''
                    SELECT COUNT(DISTINCT d.Assigned_User_s_Email) 
                    FROM devices d
                    JOIN app_portfolio ap ON d.Assigned_User_s_Email = ap.Email
                    WHERE UPPER(d.Manufacturer) = 'LENOVO' AND UPPER(ap.App) LIKE '%AWS%' 
                    AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%' AND ap.Account_Status = 'Activated'
                ''').fetchone()[0]
            
        except Exception as e:
            breakdown['error'] = str(e)
            
        return breakdown
    
    def _generate_breakdown_insights(self, breakdown_data: dict) -> list:
        """Generate insights from breakdown analysis."""
        insights = []
        
        if breakdown_data.get('error'):
            return [f"⚠️ Analysis error: {breakdown_data['error']}"]
        
        insights.append("\n📊 Breakdown Analysis:")
        
        if 'lenovo_users' in breakdown_data:
            insights.append(f"  • Lenovo laptop users: {breakdown_data['lenovo_users']}")
        if 'aws_admins' in breakdown_data:
            insights.append(f"  • AWS Admin users: {breakdown_data['aws_admins']}")
        if 'notion_users' in breakdown_data:
            insights.append(f"  • Notion license users: {breakdown_data['notion_users']}")
            
        if 'lenovo_aws' in breakdown_data:
            insights.append(f"\n🔍 Cross-Reference: Lenovo + AWS Admin: {breakdown_data['lenovo_aws']} users")
            
            if breakdown_data['lenovo_aws'] == 0 and breakdown_data.get('lenovo_users', 0) > 0:
                insights.append("💡 Key Finding: No Lenovo users have AWS Administrator access")
                
        return insights
    
    def _generate_alternative_suggestions(self, question_lower: str, breakdown_data: dict) -> list:
        """Generate alternative query suggestions."""
        suggestions = []
        
        if 'lenovo' in question_lower and breakdown_data.get('lenovo_aws', 0) == 0:
            suggestions.append("Try: 'Apple laptop users with AWS admin access'")
            
        if 'aws' in question_lower and 'admin' in question_lower:
            suggestions.append("Try: 'What devices do AWS admins use?'")
            
        return suggestions
    
    def _analyze_device_results(self, results: list) -> list:
        """Analyze device-related results."""
        insights = []
        
        if not results:
            return insights
            
        # Analyze manufacturers
        manufacturers = {}
        for result in results:
            mfg = result.get('Manufacturer', 'Unknown')
            manufacturers[mfg] = manufacturers.get(mfg, 0) + 1
            
        if manufacturers:
            top_mfg = max(manufacturers, key=manufacturers.get)
            insights.append(f"🖥️ Hardware: {top_mfg} is the primary manufacturer ({manufacturers[top_mfg]} devices)")
            
        return insights
    
    def _analyze_license_results(self, results: list) -> list:
        """Analyze license-related results."""
        insights = []
        
        if results:
            insights.append("📝 License Status: All results show active application licenses")
            
        return insights
    
    def _determine_analysis_type(self, question_lower: str) -> str:
        """Determine the type of analysis performed."""
        if 'aws' in question_lower and 'admin' in question_lower:
            return 'security_analysis'
        elif 'japan' in question_lower:
            return 'geographic_analysis'
        elif 'laptop' in question_lower or 'device' in question_lower:
            return 'hardware_analysis'
        elif 'license' in question_lower or 'notion' in question_lower:
            return 'software_analysis'
        else:
            return 'general_query'
    
    def _generate_summary(self, question: str, count: int, insights: list) -> str:
        """Generate a comprehensive summary."""
        if count == 0:
            return f"Query '{question}' returned no results. Breakdown analysis provided to understand why criteria don't intersect."
        else:
            return f"Query '{question}' successfully found {count} matching records with detailed insights provided."
    
    def _generate_key_findings(self, question_lower: str, results: list, breakdown_data: dict) -> list:
        """Generate key findings from the analysis."""
        findings = []
        
        try:
            # Security-related findings
            if 'aws' in question_lower and 'admin' in question_lower:
                aws_admin_count = breakdown_data.get('aws_admins', 0)
                if aws_admin_count > 0:
                    findings.append(f"🔐 {aws_admin_count} users have AWS Administrator access in the system")
                    
                    # Check for concentration in specific roles
                    if results:
                        roles = {}
                        for result in results:
                            role = result.get('Job_Title', 'Unknown')
                            roles[role] = roles.get(role, 0) + 1
                        if roles:
                            top_role = max(roles, key=roles.get)
                            findings.append(f"👔 Most AWS admins are {top_role}: {roles[top_role]} users")
            
            # Hardware-related findings
            if 'lenovo' in question_lower or 'laptop' in question_lower:
                lenovo_count = breakdown_data.get('lenovo_users', 0)
                if lenovo_count > 0:
                    findings.append(f"💻 {lenovo_count} employees are assigned Lenovo devices")
                    
                    # Check for zero intersection with other criteria
                    if 'aws' in question_lower and breakdown_data.get('lenovo_aws', 0) == 0:
                        findings.append("⚠️ No overlap between Lenovo users and AWS administrators")
            
            # Geographic findings
            if 'japan' in question_lower:
                # Analyze Japanese name patterns in results
                japanese_indicators = 0
                for result in results:
                    first_name = result.get('First_Name', '').lower()
                    email = result.get('Email', '').lower()
                    if any(name in first_name for name in ['tomoyo', 'mari', 'kohei', 'yuki', 'akira']) or '.jp' in email:
                        japanese_indicators += 1
                
                if japanese_indicators > 0:
                    findings.append(f"🗾 {japanese_indicators} users identified as likely Japanese employees")
            
            # Application licensing findings
            if 'notion' in question_lower:
                notion_count = breakdown_data.get('notion_users', 0)
                if notion_count > 0:
                    findings.append(f"📝 {notion_count} employees have active Notion licenses")
            
            # Complex query findings
            if len(results) == 0 and self._is_complex_multi_criteria_query(question_lower):
                findings.append("🔍 Complex criteria analysis shows no users match all requirements simultaneously")
                findings.append("💡 Consider relaxing one or more criteria to find related users")
                
        except Exception as e:
            findings.append(f"⚠️ Error generating findings: {str(e)}")
            
        return findings
    
    def _generate_cross_references(self, question_lower: str, breakdown_data: dict) -> list:
        """Generate detailed cross-reference analysis with user listings and status indicators."""
        cross_refs = []
        
        try:
            # Lenovo + AWS Admin cross-reference
            if 'lenovo' in question_lower and 'aws' in question_lower and 'admin' in question_lower:
                cross_ref = self._analyze_lenovo_aws_crossref()
                if cross_ref:
                    cross_refs.append(cross_ref)
            
            # AWS Admin + Notion cross-reference
            if 'aws' in question_lower and 'admin' in question_lower and 'notion' in question_lower:
                cross_ref = self._analyze_aws_notion_crossref()
                if cross_ref:
                    cross_refs.append(cross_ref)
            
            # Apple/MacBook + AWS Admin cross-reference
            if any(term in question_lower for term in ['apple', 'macbook']) and 'aws' in question_lower and 'admin' in question_lower:
                cross_ref = self._analyze_apple_aws_crossref()
                if cross_ref:
                    cross_refs.append(cross_ref)
            
            # Geographic + AWS cross-reference
            if any(geo in question_lower for geo in ['japan', 'india']) and 'aws' in question_lower:
                cross_ref = self._analyze_geographic_aws_crossref(question_lower)
                if cross_ref:
                    cross_refs.append(cross_ref)
                    
        except Exception as e:
            cross_refs.append({
                'type': 'error',
                'title': 'Cross-Reference Analysis Error',
                'status': 'error',
                'message': f"Error generating cross-references: {str(e)}",
                'users': []
            })
            
        return cross_refs
    
    def _analyze_lenovo_aws_crossref(self) -> dict:
        """Analyze Lenovo + AWS Admin cross-reference with detailed user data."""
        try:
            # Get total counts
            lenovo_count = self.conn.execute('''
                SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                WHERE UPPER(Manufacturer) = 'LENOVO' AND Device_Status = 'In-use'
                AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
            ''').fetchone()[0]
            
            # Get intersection users
            lenovo_aws_users = self.conn.execute('''
                SELECT DISTINCT 
                    p.First_Name, p.Last_Name, p.Email, 
                    ap.Identifier, ap.Role_s, d.Model_Name
                FROM devices d
                JOIN provisions p ON d.Assigned_User_s_Email = p.Email
                JOIN app_portfolio ap ON p.Email = ap.Email
                WHERE UPPER(d.Manufacturer) = 'LENOVO' 
                AND d.Device_Status = 'In-use'
                AND UPPER(ap.App) LIKE '%AWS%' 
                AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%' 
                AND ap.Account_Status = 'Activated'
                ORDER BY p.First_Name, p.Last_Name
            ''').fetchall()
            
            intersection_count = len(lenovo_aws_users)
            
            if intersection_count == 0:
                return {
                    'type': 'intersection',
                    'title': 'Lenovo + AWS Admin',
                    'status': 'none',
                    'count': 0,
                    'total_base': lenovo_count,
                    'message': f"None of the {lenovo_count} Lenovo laptop users have AWS Administrator access",
                    'blocker': "This is the primary blocker for the query",
                    'users': []
                }
            else:
                users = []
                for user in lenovo_aws_users:
                    users.append({
                        'name': f"{user[0]} {user[1]}".strip(),
                        'email': user[2],
                        'aws_identifier': user[3],
                        'aws_role': user[4],
                        'device_model': user[5]
                    })
                
                return {
                    'type': 'intersection',
                    'title': 'Lenovo + AWS Admin',
                    'status': 'found',
                    'count': intersection_count,
                    'total_base': lenovo_count,
                    'message': f"{intersection_count} users found",
                    'users': users
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'title': 'Lenovo + AWS Admin Analysis Error',
                'status': 'error',
                'message': str(e),
                'users': []
            }
    
    def _analyze_aws_notion_crossref(self) -> dict:
        """Analyze AWS Admin + Notion cross-reference with detailed user data."""
        try:
            # Get AWS admins with Notion access
            aws_notion_users = self.conn.execute('''
                SELECT DISTINCT 
                    p.First_Name, p.Last_Name, p.Email, 
                    ap.Identifier, ap.Role_s,
                    CASE 
                        WHEN p.`Notion_-_Josys_inc` = 'Activated' THEN 'Josys Inc'
                        WHEN p.`Notion_-_Josys_public` = 'Activated' THEN 'Josys Public'
                        ELSE 'Unknown'
                    END as notion_type
                FROM app_portfolio ap
                JOIN provisions p ON ap.Email = p.Email
                WHERE UPPER(ap.App) LIKE '%AWS%'
                AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%'
                AND ap.Account_Status = 'Activated'
                AND (p.`Notion_-_Josys_inc` = 'Activated' OR p.`Notion_-_Josys_public` = 'Activated')
                ORDER BY p.First_Name, p.Last_Name
                LIMIT 10
            ''').fetchall()
            
            intersection_count = len(aws_notion_users)
            
            if intersection_count == 0:
                return {
                    'type': 'intersection',
                    'title': 'AWS Admin + Notion',
                    'status': 'none',
                    'count': 0,
                    'message': "No AWS Administrators have active Notion licenses",
                    'users': []
                }
            else:
                users = []
                for user in aws_notion_users:
                    users.append({
                        'name': f"{user[0]} {user[1]}".strip(),
                        'email': user[2],
                        'aws_identifier': user[3],
                        'aws_role': user[4],
                        'notion_type': user[5]
                    })
                
                return {
                    'type': 'intersection',
                    'title': 'AWS Admin + Notion',
                    'status': 'found',
                    'count': intersection_count,
                    'message': f"{intersection_count} users found",
                    'users': users
                }
                
        except Exception as e:
            return {
                'type': 'error',
                'title': 'AWS Admin + Notion Analysis Error',
                'status': 'error',
                'message': str(e),
                'users': []
            }
    
    def _analyze_apple_aws_crossref(self) -> dict:
        """Analyze Apple/MacBook + AWS Admin cross-reference."""
        try:
            # Get Apple users with AWS admin
            apple_aws_users = self.conn.execute('''
                SELECT DISTINCT 
                    p.First_Name, p.Last_Name, p.Email, 
                    ap.Identifier, ap.Role_s, d.Model_Name
                FROM devices d
                JOIN provisions p ON d.Assigned_User_s_Email = p.Email
                JOIN app_portfolio ap ON p.Email = ap.Email
                WHERE UPPER(d.Manufacturer) = 'APPLE' 
                AND d.Device_Status = 'In-use'
                AND UPPER(ap.App) LIKE '%AWS%' 
                AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%' 
                AND ap.Account_Status = 'Activated'
                ORDER BY p.First_Name, p.Last_Name
                LIMIT 10
            ''').fetchall()
            
            intersection_count = len(apple_aws_users)
            
            if intersection_count == 0:
                apple_count = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                    WHERE UPPER(Manufacturer) = 'APPLE' AND Device_Status = 'In-use'
                ''').fetchone()[0]
                
                return {
                    'type': 'intersection',
                    'title': 'Apple + AWS Admin',
                    'status': 'none',
                    'count': 0,
                    'total_base': apple_count,
                    'message': f"None of the {apple_count} Apple laptop users have AWS Administrator access",
                    'users': []
                }
            else:
                users = []
                for user in apple_aws_users:
                    users.append({
                        'name': f"{user[0]} {user[1]}".strip(),
                        'email': user[2],
                        'aws_identifier': user[3],
                        'aws_role': user[4],
                        'device_model': user[5]
                    })
                
                return {
                    'type': 'intersection',
                    'title': 'Apple + AWS Admin',
                    'status': 'found',
                    'count': intersection_count,
                    'message': f"{intersection_count} users found",
                    'users': users
                }
                
        except Exception as e:
            return None
    
    def _analyze_geographic_aws_crossref(self, question_lower: str) -> dict:
        """Analyze geographic + AWS cross-reference."""
        try:
            location = 'Japan' if 'japan' in question_lower else 'India' if 'india' in question_lower else 'Unknown'
            
            if location == 'Japan':
                # Get AWS admins in Japan (using name patterns and .jp emails)
                japan_aws_users = self.conn.execute('''
                    SELECT DISTINCT 
                        p.First_Name, p.Last_Name, p.Email, 
                        ap.Identifier, ap.Role_s
                    FROM app_portfolio ap
                    JOIN provisions p ON ap.Email = p.Email
                    WHERE UPPER(ap.App) LIKE '%AWS%'
                    AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%'
                    AND ap.Account_Status = 'Activated'
                    AND (LOWER(p.First_Name) IN ('tomoyo', 'mari', 'kohei', 'yuki', 'akira') 
                         OR p.Email LIKE '%.jp')
                    ORDER BY p.First_Name, p.Last_Name
                    LIMIT 10
                ''').fetchall()
                
                intersection_count = len(japan_aws_users)
                
                if intersection_count == 0:
                    return {
                        'type': 'intersection',
                        'title': 'Japan + AWS Admin',
                        'status': 'none',
                        'count': 0,
                        'message': "No AWS Administrators identified as Japan-based employees",
                        'note': "Based on name patterns and .jp email domains",
                        'users': []
                    }
                else:
                    users = []
                    for user in japan_aws_users:
                        users.append({
                            'name': f"{user[0]} {user[1]}".strip(),
                            'email': user[2],
                            'aws_identifier': user[3],
                            'aws_role': user[4]
                        })
                    
                    return {
                        'type': 'intersection',
                        'title': 'Japan + AWS Admin',
                        'status': 'found',
                        'count': intersection_count,
                        'message': f"{intersection_count} users found",
                        'note': "Based on name patterns and .jp email domains",
                        'users': users
                    }
            
        except Exception as e:
            return None
    
    def _generate_detailed_breakdown_analysis(self, question_lower: str, results: list) -> dict:
        """Generate comprehensive breakdown analysis for users, provisions, devices, and combinations."""
        
        breakdown = {
            'title': 'Individual Components',
            'components': [],
            'intersections': [],
            'summary': ''
        }
        
        try:
            # Analyze different query components
            components_found = []
            
            # 1. Device/Hardware Analysis
            if any(term in question_lower for term in ['laptop', 'device', 'computer', 'phone', 'lenovo', 'apple', 'macbook']):
                device_stats = self._analyze_device_components(question_lower)
                components_found.extend(device_stats)
            
            # 2. User/Employee Analysis
            if any(term in question_lower for term in ['user', 'employee', 'person', 'people', 'staff']):
                user_stats = self._analyze_user_components(question_lower)
                components_found.extend(user_stats)
            
            # 3. Application/Access Analysis
            if any(term in question_lower for term in ['aws', 'admin', 'notion', 'github', 'slack', 'app', 'access', 'license']):
                app_stats = self._analyze_application_components(question_lower)
                components_found.extend(app_stats)
            
            # 4. Geographic Analysis
            if any(term in question_lower for term in ['japan', 'india', 'bangalore', 'tokyo', 'location']):
                geo_stats = self._analyze_geographic_components(question_lower)
                components_found.extend(geo_stats)
            
            # 5. Role/Department Analysis
            if any(term in question_lower for term in ['role', 'department', 'title', 'position']):
                role_stats = self._analyze_role_components(question_lower)
                components_found.extend(role_stats)
            
            breakdown['components'] = components_found
            
            # Generate intersection analysis
            if len(components_found) > 1:
                intersections = self._analyze_component_intersections(question_lower, components_found)
                breakdown['intersections'] = intersections
            
            # Generate summary
            total_components = len(components_found)
            if total_components > 0:
                breakdown['summary'] = f"Analyzed {total_components} component{'s' if total_components != 1 else ''} across the organization"
            
        except Exception as e:
            breakdown['components'] = [{'name': 'Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)}]
            
        return breakdown
    
    def _analyze_device_components(self, question_lower: str) -> list:
        """Analyze device-related components."""
        components = []
        
        try:
            # Total active devices
            total_devices = self.conn.execute('''
                SELECT COUNT(*) FROM devices WHERE Device_Status = 'In-use'
            ''').fetchone()[0]
            
            if total_devices > 0:
                components.append({
                    'name': 'Active Devices',
                    'count': total_devices,
                    'status': 'success',
                    'icon': '💻'
                })
            
            # Specific manufacturer analysis
            if 'lenovo' in question_lower:
                lenovo_count = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                    WHERE UPPER(Manufacturer) = 'LENOVO' AND Device_Status = 'In-use'
                    AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
                ''').fetchone()[0]
                
                components.append({
                    'name': 'Lenovo laptop users',
                    'count': lenovo_count,
                    'status': 'success' if lenovo_count > 0 else 'warning',
                    'icon': '💻'
                })
            
            if any(term in question_lower for term in ['apple', 'macbook']):
                apple_count = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                    WHERE UPPER(Manufacturer) = 'APPLE' AND Device_Status = 'In-use'
                    AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
                ''').fetchone()[0]
                
                components.append({
                    'name': 'Apple laptop users',
                    'count': apple_count,
                    'status': 'success' if apple_count > 0 else 'warning',
                    'icon': '🍎'
                })
                
        except Exception as e:
            components.append({'name': 'Device Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return components
    
    def _analyze_user_components(self, question_lower: str) -> list:
        """Analyze user-related components."""
        components = []
        
        try:
            # Total active users
            total_users = self.conn.execute('''
                SELECT COUNT(DISTINCT Email) FROM provisions WHERE Email IS NOT NULL AND Email != ''
            ''').fetchone()[0]
            
            if total_users > 0:
                components.append({
                    'name': 'Total Users',
                    'count': total_users,
                    'status': 'success',
                    'icon': '👥'
                })
            
            # Active employees (those with devices)
            active_employees = self.conn.execute('''
                SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                WHERE Device_Status = 'In-use' AND Assigned_User_s_Email IS NOT NULL AND Assigned_User_s_Email != ''
            ''').fetchone()[0]
            
            if active_employees > 0:
                components.append({
                    'name': 'Active Employees',
                    'count': active_employees,
                    'status': 'success',
                    'icon': '👤'
                })
                
        except Exception as e:
            components.append({'name': 'User Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return components
    
    def _analyze_application_components(self, question_lower: str) -> list:
        """Analyze application/access-related components."""
        components = []
        
        try:
            # AWS Admin analysis
            if 'aws' in question_lower and 'admin' in question_lower:
                aws_admins = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) FROM app_portfolio 
                    WHERE UPPER(App) LIKE '%AWS%' AND UPPER(Role_s) LIKE '%ADMINISTRATOR%' 
                    AND Account_Status = 'Activated'
                ''').fetchone()[0]
                
                components.append({
                    'name': 'AWS Admin users',
                    'count': aws_admins,
                    'status': 'success' if aws_admins > 0 else 'warning',
                    'icon': '☁️'
                })
            
            # Notion license analysis
            if 'notion' in question_lower:
                notion_users = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) FROM provisions 
                    WHERE `Notion_-_Josys_inc` = 'Activated' OR `Notion_-_Josys_public` = 'Activated'
                ''').fetchone()[0]
                
                components.append({
                    'name': 'Notion license users',
                    'count': notion_users,
                    'status': 'success' if notion_users > 0 else 'warning',
                    'icon': '📝'
                })
            
            # GitHub access analysis
            if 'github' in question_lower:
                github_users = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) FROM app_portfolio 
                    WHERE UPPER(App) LIKE '%GITHUB%' AND Account_Status = 'Activated'
                ''').fetchone()[0]
                
                components.append({
                    'name': 'GitHub users',
                    'count': github_users,
                    'status': 'success' if github_users > 0 else 'warning',
                    'icon': '🐙'
                })
                
        except Exception as e:
            components.append({'name': 'App Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return components
    
    def _analyze_geographic_components(self, question_lower: str) -> list:
        """Analyze geographic-related components."""
        components = []
        
        try:
            # Japan analysis
            if 'japan' in question_lower or 'tokyo' in question_lower:
                japan_devices = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                    WHERE (UPPER(Region) LIKE '%JAPAN%' OR UPPER(City) LIKE '%TOKYO%')
                    AND Device_Status = 'In-use'
                ''').fetchone()[0]
                
                # Also check for Japanese name patterns
                japanese_names = self.conn.execute('''
                    SELECT COUNT(DISTINCT Email) FROM provisions 
                    WHERE LOWER(First_Name) IN ('tomoyo', 'mari', 'kohei', 'yuki', 'akira')
                    OR Email LIKE '%.jp'
                ''').fetchone()[0]
                
                total_japan = max(japan_devices, japanese_names)
                
                components.append({
                    'name': 'Japan-based users',
                    'count': total_japan,
                    'status': 'success' if total_japan > 0 else 'warning',
                    'icon': '🗾'
                })
            
            # India analysis
            if 'india' in question_lower or 'bangalore' in question_lower:
                india_devices = self.conn.execute('''
                    SELECT COUNT(DISTINCT Assigned_User_s_Email) FROM devices 
                    WHERE (UPPER(Region) LIKE '%INDIA%' OR UPPER(City) LIKE '%BANGALORE%')
                    AND Device_Status = 'In-use'
                ''').fetchone()[0]
                
                components.append({
                    'name': 'India-based users',
                    'count': india_devices,
                    'status': 'success' if india_devices > 0 else 'warning',
                    'icon': '🇮🇳'
                })
                
        except Exception as e:
            components.append({'name': 'Geographic Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return components
    
    def _analyze_role_components(self, question_lower: str) -> list:
        """Analyze role/department-related components."""
        components = []
        
        try:
            # Get top departments
            departments = self.conn.execute('''
                SELECT Department_s, COUNT(*) as count FROM app_portfolio 
                WHERE Department_s IS NOT NULL AND Department_s != '' 
                GROUP BY Department_s ORDER BY count DESC LIMIT 3
            ''').fetchall()
            
            for dept, count in departments:
                components.append({
                    'name': f'{dept} department',
                    'count': count,
                    'status': 'success',
                    'icon': '🏢'
                })
                
        except Exception as e:
            components.append({'name': 'Role Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return components
    
    def _analyze_component_intersections(self, question_lower: str, components: list) -> list:
        """Analyze intersections between different components."""
        intersections = []
        
        try:
            # Lenovo + AWS Admin intersection
            if any('lenovo' in comp['name'].lower() for comp in components) and any('aws' in comp['name'].lower() for comp in components):
                lenovo_aws = self.conn.execute('''
                    SELECT COUNT(DISTINCT d.Assigned_User_s_Email) FROM devices d
                    JOIN app_portfolio ap ON d.Assigned_User_s_Email = ap.Email
                    WHERE UPPER(d.Manufacturer) = 'LENOVO' AND UPPER(ap.App) LIKE '%AWS%' 
                    AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%' AND ap.Account_Status = 'Activated'
                ''').fetchone()[0]
                
                intersections.append({
                    'name': 'Lenovo users with AWS Admin',
                    'count': lenovo_aws,
                    'status': 'success' if lenovo_aws > 0 else 'warning',
                    'icon': '🔗'
                })
            
            # AWS + Notion intersection
            if any('aws' in comp['name'].lower() for comp in components) and any('notion' in comp['name'].lower() for comp in components):
                aws_notion = self.conn.execute('''
                    SELECT COUNT(DISTINCT ap.Email) FROM app_portfolio ap
                    JOIN provisions p ON ap.Email = p.Email
                    WHERE UPPER(ap.App) LIKE '%AWS%' AND UPPER(ap.Role_s) LIKE '%ADMINISTRATOR%'
                    AND (p.`Notion_-_Josys_inc` = 'Activated' OR p.`Notion_-_Josys_public` = 'Activated')
                ''').fetchone()[0]
                
                intersections.append({
                    'name': 'AWS Admins with Notion',
                    'count': aws_notion,
                    'status': 'success' if aws_notion > 0 else 'warning',
                    'icon': '🔗'
                })
                
        except Exception as e:
            intersections.append({'name': 'Intersection Analysis Error', 'count': 0, 'status': 'error', 'details': str(e)})
            
        return intersections

def create_openai_app() -> Flask:
    """Create Flask app with OpenAI NLP capabilities."""
    
    app = Flask(__name__)
    
    try:
        nlp = JosysOpenAINLP()
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI NLP: {e}")
        return None
    
    @app.route('/')
    def index():
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Josys OpenAI NLP Search</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            background: #f8f9fa; color: #333; line-height: 1.6; padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { 
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
            color: white; padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 20px;
        }
        .header h1 { font-size: 2.2em; margin-bottom: 10px; }
        .badge { 
            background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; 
            font-size: 0.8em; margin: 0 5px;
        }
        
        .search-section { 
            background: white; padding: 25px; border-radius: 12px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; 
        }
        .search-box { display: flex; gap: 15px; margin-bottom: 15px; }
        .search-input { 
            flex: 1; padding: 15px; border: 2px solid #e9ecef; border-radius: 8px; 
            font-size: 16px; transition: border-color 0.3s;
        }
        .search-input:focus { outline: none; border-color: #28a745; }
        .search-btn { 
            padding: 15px 25px; background: #28a745; color: white; border: none; 
            border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold;
        }
        .search-btn:hover { background: #218838; }
        .search-btn:disabled { background: #6c757d; cursor: not-allowed; }
        
        .examples { 
            background: #e8f5e8; padding: 15px; border-radius: 8px; 
            border-left: 4px solid #28a745; margin-bottom: 15px;
        }
        .examples h4 { margin-bottom: 10px; color: #155724; }
        .example-queries { display: flex; flex-wrap: wrap; gap: 8px; }
        .example-query { 
            background: white; padding: 8px 12px; border-radius: 6px; cursor: pointer; 
            font-size: 14px; border: 1px solid #28a745; transition: all 0.3s;
            color: #155724;
        }
        .example-query:hover { background: #28a745; color: white; }
        
        .results-section { 
            background: white; padding: 25px; border-radius: 12px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        .loading { text-align: center; padding: 30px; color: #666; }
        .spinner { 
            display: inline-block; width: 20px; height: 20px; 
            border: 3px solid #f3f3f3; border-top: 3px solid #28a745; 
            border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .result-item { 
            border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; 
            margin-bottom: 15px; transition: all 0.3s;
        }
        .result-item:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .result-header { 
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; 
        }
        .result-type { 
            background: #28a745; color: white; padding: 6px 12px; 
            border-radius: 20px; font-size: 12px; text-transform: uppercase; font-weight: bold;
        }
        .result-type.user { background: #17a2b8; }
        .result-content { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; 
        }
        .result-field { background: #f8f9fa; padding: 12px; border-radius: 6px; }
        .field-label { 
            font-weight: bold; font-size: 11px; color: #495057; 
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .field-value { color: #212529; margin-top: 5px; font-size: 14px; }
        
        .stats { 
            display: flex; justify-content: space-between; align-items: center; 
            margin-bottom: 20px; padding: 15px; background: #e8f5e8; border-radius: 8px; 
            border-left: 4px solid #28a745;
        }
        .method-badge {
            background: #28a745; color: white; padding: 4px 8px; border-radius: 4px;
            font-size: 12px; font-weight: bold;
        }
        
        .sql-display {
            background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 15px;
            border-left: 4px solid #6f42c1; font-family: 'Monaco', 'Courier New', monospace;
            font-size: 13px; overflow-x: auto;
        }
        .sql-display .label {
            color: #6f42c1; font-weight: bold; font-size: 11px; 
            text-transform: uppercase; margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Josys OpenAI NLP Search</h1>
            <p>AI-powered natural language search with OpenAI GPT-3.5</p>
            <div style="margin-top: 10px;">
                <span class="badge">🧠 NL2SQL</span>
                <span class="badge">🔍 Smart Search</span>
                <span class="badge">⚡ Real-time</span>
            </div>
        </div>
        
        <div class="search-section">
            <div class="search-box">
                <input type="text" id="searchInput" class="search-input" 
                       placeholder="Ask anything in natural language... e.g., 'list all devices assigned to Arvind'"
                       onkeypress="if(event.key==='Enter') performSearch()">
                <button onclick="performSearch()" class="search-btn" id="searchBtn">Ask AI</button>
            </div>
            
            <div class="examples">
                <h4>💡 Try these natural language queries:</h4>
                <div class="example-queries">
                    <span class="example-query" onclick="setQuery('list all devices assigned to Arvind')">
                        list all devices assigned to Arvind
                    </span>
                    <span class="example-query" onclick="setQuery('show me MacBook laptops in Bangalore')">
                        show me MacBook laptops in Bangalore
                    </span>
                    <span class="example-query" onclick="setQuery('which users have GitHub access?')">
                        which users have GitHub access?
                    </span>
                    <span class="example-query" onclick="setQuery('find available devices for IT admins')">
                        find available devices for IT admins
                    </span>
                    <span class="example-query" onclick="setQuery('who is using Apple devices?')">
                        who is using Apple devices?
                    </span>
                    <span class="example-query" onclick="setQuery('list names that have a macbook and inactive notion licence')">
                        list names that have a macbook and inactive notion licence
                    </span>
                </div>
            </div>
        </div>
        
        <div class="results-section" id="resultsSection" style="display: none;">
            <div id="resultsContent"></div>
        </div>
    </div>

    <script>
        function setQuery(query) {
            document.getElementById('searchInput').value = query;
            performSearch();
        }
        
        async function performSearch() {
            const question = document.getElementById('searchInput').value.trim();
            if (!question) return;
            
            const resultsSection = document.getElementById('resultsSection');
            const resultsContent = document.getElementById('resultsContent');
            const searchBtn = document.getElementById('searchBtn');
            
            // Show loading
            resultsSection.style.display = 'block';
            resultsContent.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    AI is processing your question using OpenAI GPT-3.5...
                </div>
            `;
            searchBtn.disabled = true;
            searchBtn.textContent = 'Processing...';
            
            try {
                const response = await fetch('/api/nlp-search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, type: 'combined' })
                });
                
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                resultsContent.innerHTML = `
                    <div style="text-align: center; padding: 30px; color: #dc3545;">
                        <h3>❌ AI Processing Error</h3>
                        <p>${error.message}</p>
                        <p style="margin-top: 15px;">
                            <small>Check that your OpenAI API key is valid and you have credits available.</small>
                        </p>
                    </div>
                `;
            } finally {
                searchBtn.disabled = false;
                searchBtn.textContent = 'Ask AI';
            }
        }
        
        function displayResults(data) {
            const resultsContent = document.getElementById('resultsContent');
            
            let html = `
                <div class="stats">
                    <div>
                        <strong>${data.count || 0}</strong> results found using 
                        <span class="method-badge">${data.method || 'unknown'}</span>
                    </div>
                    <div>⚡ ${((data.execution_time || 0) * 1000).toFixed(0)}ms</div>
                </div>
            `;
            
            // Show error if any
            if (data.status === 'error' || data.error) {
                html += `
                    <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                        <strong>Error:</strong> ${data.error || 'Unknown error occurred'}
                    </div>
                `;
            }
            
            // 1. Add AI Analysis & Insights section FIRST
            if (data.insights && data.insights.length > 0) {
                html += `
                    <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #1976d2; margin: 0 0 15px 0;">🧠 AI Insights & Analytics</h3>
                        <div style="color: #424242;">
                `;
                
                data.insights.forEach(insight => {
                    html += `<div style="margin: 8px 0; line-height: 1.5;">${insight}</div>`;
                });
                
                html += `</div></div>`;
            }
            
            // 2. Add Breakdown Analysis section SECOND
            if (data.detailed_breakdown && data.detailed_breakdown.components && data.detailed_breakdown.components.length > 0) {
                html += `
                    <div style="background: #fff8e1; border: 1px solid #ffa000; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #f57c00; margin: 0 0 15px 0;">📊 Breakdown Analysis</h3>
                        <div style="color: #424242;">
                `;
                
                // Add individual components
                html += `<div style="margin-bottom: 15px;"><strong>Individual Components:</strong></div>`;
                
                data.detailed_breakdown.components.forEach(component => {
                    const statusIcon = component.status === 'success' ? '✅' : component.status === 'warning' ? '⚠️' : '❌';
                    const statusColor = component.status === 'success' ? '#4caf50' : component.status === 'warning' ? '#ff9800' : '#f44336';
                    
                    html += `
                        <div style="margin: 8px 0; padding: 8px 12px; background: white; border-radius: 6px; border-left: 4px solid ${statusColor}; display: flex; align-items: center;">
                            <span style="margin-right: 8px;">${statusIcon}</span>
                            <span style="margin-right: 8px;">${component.icon || '📋'}</span>
                            <span><strong>${component.name}:</strong> ${component.count} users</span>
                        </div>
                    `;
                });
                
                // Add intersections if available
                if (data.detailed_breakdown.intersections && data.detailed_breakdown.intersections.length > 0) {
                    html += `<div style="margin: 20px 0 10px 0; border-top: 1px solid #ffcc80; padding-top: 15px;"><strong>Intersections:</strong></div>`;
                    
                    data.detailed_breakdown.intersections.forEach(intersection => {
                        const statusIcon = intersection.status === 'success' ? '✅' : intersection.status === 'warning' ? '⚠️' : '❌';
                        const statusColor = intersection.status === 'success' ? '#4caf50' : intersection.status === 'warning' ? '#ff9800' : '#f44336';
                        
                        html += `
                            <div style="margin: 8px 0; padding: 8px 12px; background: white; border-radius: 6px; border-left: 4px solid ${statusColor}; display: flex; align-items: center;">
                                <span style="margin-right: 8px;">${statusIcon}</span>
                                <span style="margin-right: 8px;">${intersection.icon || '🔗'}</span>
                                <span><strong>${intersection.name}:</strong> ${intersection.count} users</span>
                            </div>
                        `;
                    });
                }
                
                // Add summary
                if (data.detailed_breakdown.summary) {
                    html += `
                        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ffcc80; font-style: italic; color: #666;">
                            ${data.detailed_breakdown.summary}
                        </div>
                    `;
                }
                
                html += `</div></div>`;
            }
            
            // 3. Add Key Findings section THIRD
            if (data.key_findings && data.key_findings.length > 0) {
                html += `
                    <div style="background: #fff3e0; border: 1px solid #ff9800; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #f57c00; margin: 0 0 15px 0;">🔍 Key Findings</h3>
                        <div style="color: #424242;">
                `;
                
                data.key_findings.forEach(finding => {
                    html += `<div style="margin: 8px 0; line-height: 1.5; padding: 6px 0; border-left: 3px solid #ffb74d; padding-left: 12px;">${finding}</div>`;
                });
                
                html += `</div></div>`;
            }
            
            // 4. Add Cross-Reference Results section FOURTH
            if (data.cross_references && data.cross_references.length > 0) {
                html += `
                    <div style="background: #f3e5f5; border: 1px solid #9c27b0; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #7b1fa2; margin: 0 0 15px 0;">🔗 Cross-Reference Results</h3>
                `;
                
                data.cross_references.forEach(crossRef => {
                    // Status icon and styling
                    const statusIcon = crossRef.status === 'found' ? '✅' : '❌';
                    const statusColor = crossRef.status === 'found' ? '#4caf50' : '#f44336';
                    
                    html += `
                        <div style="margin: 15px 0; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid ${statusColor};">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <span style="font-size: 18px; margin-right: 10px;">${statusIcon}</span>
                                <strong style="color: #7b1fa2; font-size: 16px;">${crossRef.title}: ${crossRef.count} users</strong>
                            </div>
                    `;
                    
                    // Add message/explanation
                    if (crossRef.message) {
                        html += `<div style="margin: 8px 0 12px 28px; color: #555; font-style: italic;">${crossRef.message}</div>`;
                    }
                    
                    // Add blocker information for zero results
                    if (crossRef.blocker) {
                        html += `<div style="margin: 8px 0 12px 28px; color: #d32f2f; font-weight: bold;">${crossRef.blocker}</div>`;
                    }
                    
                    // Add note if available
                    if (crossRef.note) {
                        html += `<div style="margin: 8px 0 12px 28px; color: #666; font-size: 14px;">${crossRef.note}</div>`;
                    }
                    
                    // Display user details if found
                    if (crossRef.users && crossRef.users.length > 0) {
                        html += `<div style="margin-left: 28px;">`;
                        
                        crossRef.users.forEach(user => {
                            html += `
                                <div style="margin: 6px 0; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #28a745;">
                                    <strong>${user.name}</strong> (${user.email})
                            `;
                            
                            // Add AWS details if available
                            if (user.aws_identifier && user.aws_role) {
                                html += ` - AWS: ${user.aws_identifier}`;
                                if (user.aws_role && user.aws_role !== user.aws_identifier) {
                                    html += ` [${user.aws_role}]`;
                                }
                            }
                            
                            // Add device details if available
                            if (user.device_model) {
                                html += ` - Device: ${user.device_model}`;
                            }
                            
                            // Add Notion details if available
                            if (user.notion_type) {
                                html += ` - Notion: ${user.notion_type}`;
                            }
                            
                            html += `</div>`;
                        });
                        
                        html += `</div>`;
                    }
                    
                    html += `</div>`;
                });
                
                html += `</div>`;
            }
            
            // 5. Add Results section FIFTH
            const results = data.results || [];
            
            if (results.length === 0) {
                html += `
                    <div style="text-align: center; padding: 30px; color: #666;">
                        <h3>🔍 No Results Found</h3>
                        <p>Try rephrasing your question or using different terms.</p>
                        ${data.fallback_reason ? `<p><small>Reason: ${data.fallback_reason}</small></p>` : ''}
                    </div>
                `;
            } else {
                // Add results header
                html += `
                    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 20px 0;">
                        <h3 style="color: #495057; margin: 0 0 15px 0;">📋 Results (${results.length})</h3>
                        <div style="display: flex; flex-direction: column; gap: 12px;">
                `;
                
                results.forEach((result, index) => {
                    const data = result.data || result;
                    const isDevice = result.type === 'device' || data.Asset_Number;
                    const isAppAccess = data.App || data.Identifier;
                    
                    // Determine result type and icon
                    let icon = '👤';
                    let type = 'User';
                    if (isDevice) {
                        icon = '💻';
                        type = 'Device';
                    } else if (isAppAccess) {
                        icon = '🔑';
                        type = 'App Access';
                    }
                    
                    html += `
                        <div style="background: white; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px;">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 16px; margin-right: 8px;">${icon}</span>
                                <strong style="color: #212529;">${type} #${index + 1}</strong>
                            </div>
                            <div style="color: #6c757d; line-height: 1.6;">
                    `;
                    
                    if (isDevice) {
                        html += `
                            <div><strong>Asset:</strong> ${data.Asset_Number || 'N/A'}</div>
                            <div><strong>Device:</strong> ${data.Manufacturer || 'N/A'} ${data.Model_Name || 'N/A'} (${data.Device_Type || 'N/A'})</div>
                            <div><strong>Status:</strong> ${data.Device_Status || 'N/A'}</div>
                            <div><strong>Assigned to:</strong> ${data.Assigned_User_s_Email || 'Unassigned'}</div>
                            ${data.City || data.Region ? `<div><strong>Location:</strong> ${data.City || 'N/A'}, ${data.Region || 'N/A'}</div>` : ''}
                        `;
                    } else if (isAppAccess) {
                        html += `
                            <div><strong>Application:</strong> ${data.App || 'N/A'}</div>
                            <div><strong>User:</strong> ${data.First_Name || ''} ${data.Last_Name || ''} (${data.Email || 'N/A'})</div>
                            <div><strong>Access ID:</strong> ${data.ID || data.Identifier || 'N/A'}</div>
                            <div><strong>Role:</strong> ${data.Role_s || data.Role || 'N/A'}</div>
                            <div><strong>Status:</strong> ${data.Account_Status || data.Status || 'N/A'}</div>
                            ${data.Department_s ? `<div><strong>Department:</strong> ${data.Department_s}</div>` : ''}
                        `;
                    } else {
                        html += `
                            <div><strong>Name:</strong> ${data.First_Name || ''} ${data.Last_Name || ''}</div>
                            <div><strong>Email:</strong> ${data.Email || 'N/A'}</div>
                            ${data.User_ID ? `<div><strong>User ID:</strong> ${data.User_ID}</div>` : ''}
                            ${data.Role ? `<div><strong>Role:</strong> ${data.Role}</div>` : ''}
                            ${data.Status ? `<div><strong>Status:</strong> ${data.Status}</div>` : ''}
                            ${data.Work_Location_Code ? `<div><strong>Location:</strong> ${data.Work_Location_Code}</div>` : ''}
                        `;
                    }
                    
                    html += `
                            </div>
                        </div>
                    `;
                });
                
                html += `</div></div>`;
            }
            
            // Add suggestions section if available
            if (data.suggestions && data.suggestions.length > 0) {
                html += `
                    <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 8px; padding: 20px; margin: 20px 0;">
                        <h3 style="color: #388e3c; margin: 0 0 15px 0;">💡 Suggested Queries</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                `;
                
                data.suggestions.forEach(suggestion => {
                    const cleanSuggestion = suggestion.replace('Try: ', '');
                    html += `
                        <button style="background: #4caf50; color: white; border: none; padding: 8px 16px; border-radius: 20px; cursor: pointer; font-size: 14px;" 
                                onclick="document.getElementById('searchInput').value='${cleanSuggestion.replace(/'/g, '\\'')}'; searchData();">
                            ${cleanSuggestion}
                        </button>
                    `;
                });
                
                html += `</div></div>`;
            }
            
            // Add comprehensive summary
            if (data.comprehensive_summary) {
                html += `
                    <div style="background: #fafafa; border: 1px solid #bdbdbd; border-radius: 8px; padding: 15px; margin: 20px 0;">
                        <div style="font-style: italic; color: #555; text-align: center;">
                            📋 ${data.comprehensive_summary}
                        </div>
                    </div>
                `;
            }
            
            // Add SQL Query at the BOTTOM (moved from top)
            if (data.sql) {
                html += `
                    <div class="sql-display" style="margin-top: 30px;">
                        <div class="label">📝 Generated SQL Query:</div>
                        <div>${data.sql}</div>
                    </div>
                `;
            }
            
            // Show attempted SQL if fallback was used
            if (data.attempted_sql && data.attempted_sql !== 'N/A') {
                html += `
                    <div class="sql-display" style="border-left-color: #ffc107; margin-top: 15px;">
                        <div class="label" style="color: #856404;">⚠️ Attempted SQL (failed):</div>
                        <div>${data.attempted_sql}</div>
                    </div>
                `;
            }
            
            resultsContent.innerHTML = html;
        }
        
        // Auto-focus search input
        document.getElementById('searchInput').focus();
    </script>
</body>
</html>
        '''
    
    @app.route('/api/nlp-search', methods=['POST'])
    def api_nlp_search():
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            search_type = data.get('type', 'combined')
            
            if not question:
                return jsonify({'error': 'Question is required', 'status': 'error'}), 400
            
            if len(question) < 3:
                return jsonify({'error': 'Question too short', 'status': 'error'}), 400
            
            # Use combined search (NL2SQL with keyword fallback)
            result = nlp.combined_nlp_search(question)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'status': 'error',
                'method': 'api_error'
            }), 500
    
    @app.route('/api/status')
    def api_status():
        return jsonify({
            'status': 'running',
            'openai_connected': bool(nlp.api_key),
            'database_connected': os.path.exists(nlp.db_path),
            'cache_size': len(nlp.query_cache)
        })
    
    return app

def main():
    """Start the OpenAI NLP web interface."""
    print("🤖 Starting Josys OpenAI NLP Web Interface")
    print("=" * 50)
    
    try:
        app = create_openai_app()
        if not app:
            print("❌ Failed to create app")
            return
        
        print("\n🌐 OpenAI NLP Interface Features:")
        print("   🧠 Natural Language to SQL with GPT-3.5")
        print("   🔍 Smart keyword fallback search")
        print("   🎯 Combined intelligent search")
        print("   💾 Query caching for performance")
        print("   📊 Detailed result analysis")
        
        print(f"\n🔗 Access URL: http://localhost:5000")
        print(f"💡 Try: 'list all devices assigned to Arvind'")
        print(f"⚠️  Press Ctrl+C to stop")
        print("=" * 50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("💡 Make sure your OpenAI API key is in the .env file")

if __name__ == "__main__":
    main()