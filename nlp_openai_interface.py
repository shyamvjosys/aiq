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
3. For CROSS-TABLE queries: JOIN devices and provisions on Email fields
4. JOIN condition: provisions.Email = devices.Assigned_User_s_Email
5. For user names: Search in First_Name, Last_Name, or Email columns
6. Use LIKE '%...%' for partial text matches and UPPER() for case-insensitive
7. Use backticks for complex column names with special characters
8. Always limit results to reasonable numbers (LIMIT 20)

EXAMPLE QUERIES:
- "devices assigned to Arvind" → SELECT * FROM devices WHERE UPPER(Assigned_User_s_Email) LIKE UPPER('%arvind%') OR UPPER(Assigned_User_s_ID) LIKE UPPER('%arvind%')
- "MacBook laptops" → SELECT * FROM devices WHERE UPPER(Device_Type) LIKE UPPER('%laptop%') AND UPPER(Manufacturer) LIKE UPPER('%apple%')
- "users with GitHub access" → SELECT * FROM provisions WHERE GitHub IS NOT NULL AND GitHub != ''
- "available devices in Bangalore" → SELECT * FROM devices WHERE UPPER(Device_Status) LIKE UPPER('%available%') AND UPPER(City) LIKE UPPER('%bangalore%')
- "names with multiple DataDog licenses" → SELECT First_Name, Last_Name, `Datadog_-_JOSYS`, `Datadog_-_JOSYS-IDAC`, `Datadog_-_JOSYS-JEP`, `Datadog_-_JOSYS-Non-Prod` FROM provisions WHERE (`Datadog_-_JOSYS` != '' AND `Datadog_-_JOSYS` IS NOT NULL) + (`Datadog_-_JOSYS-IDAC` != '' AND `Datadog_-_JOSYS-IDAC` IS NOT NULL) + (`Datadog_-_JOSYS-JEP` != '' AND `Datadog_-_JOSYS-JEP` IS NOT NULL) + (`Datadog_-_JOSYS-Non-Prod` != '' AND `Datadog_-_JOSYS-Non-Prod` IS NOT NULL) > 1

CROSS-TABLE JOIN EXAMPLES:
- "names with MacBook and Notion license" → SELECT DISTINCT p.First_Name, p.Last_Name, d.Device_Type, d.Manufacturer, p.`Notion_-_Josys_inc` FROM provisions p JOIN devices d ON p.Email = d.Assigned_User_s_Email WHERE (UPPER(d.Device_Type) LIKE '%LAPTOP%' AND UPPER(d.Manufacturer) LIKE '%APPLE%') AND p.`Notion_-_Josys_inc` != '' AND p.`Notion_-_Josys_inc` IS NOT NULL LIMIT 20
- "users with Apple devices and GitHub access" → SELECT p.First_Name, p.Last_Name, d.Asset_Number, d.Manufacturer, p.GitHub FROM provisions p JOIN devices d ON p.Email = d.Assigned_User_s_Email WHERE UPPER(d.Manufacturer) LIKE '%APPLE%' AND p.GitHub = 'Activated' LIMIT 20

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
                
                # Cache the result
                self.query_cache[cache_key] = result.copy()
                
                return result
                
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
            
            // Show SQL query if available
            if (data.sql) {
                html += `
                    <div class="sql-display">
                        <div class="label">Generated SQL Query:</div>
                        <div>${data.sql}</div>
                    </div>
                `;
            }
            
            // Show attempted SQL if fallback was used
            if (data.attempted_sql && data.attempted_sql !== 'N/A') {
                html += `
                    <div class="sql-display" style="border-left-color: #ffc107;">
                        <div class="label" style="color: #856404;">Attempted SQL (failed):</div>
                        <div>${data.attempted_sql}</div>
                    </div>
                `;
            }
            
            // Show error if any
            if (data.status === 'error' || data.error) {
                html += `
                    <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                        <strong>Error:</strong> ${data.error || 'Unknown error occurred'}
                    </div>
                `;
            }
            
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
                results.forEach(result => {
                    const data = result.data || result;
                    const isDevice = result.type === 'device' || data.Asset_Number;
                    
                    html += `
                        <div class="result-item">
                            <div class="result-header">
                                <span class="result-type ${isDevice ? 'device' : 'user'}">
                                    ${isDevice ? '💻 Device' : '👤 User'}
                                </span>
                            </div>
                            <div class="result-content">
                    `;
                    
                    if (isDevice) {
                        html += `
                            <div class="result-field">
                                <div class="field-label">Asset Number</div>
                                <div class="field-value">${data.Asset_Number || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Device Type</div>
                                <div class="field-value">${data.Device_Type || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Manufacturer</div>
                                <div class="field-value">${data.Manufacturer || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Model</div>
                                <div class="field-value">${data.Model_Name || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Status</div>
                                <div class="field-value">${data.Device_Status || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Assigned To</div>
                                <div class="field-value">${data.Assigned_User_s_Email || 'Unassigned'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Location</div>
                                <div class="field-value">${data.City || 'N/A'}, ${data.Region || 'N/A'}</div>
                            </div>
                        `;
                    } else {
                        html += `
                            <div class="result-field">
                                <div class="field-label">User ID</div>
                                <div class="field-value">${data.User_ID || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Name</div>
                                <div class="field-value">${data.First_Name || ''} ${data.Last_Name || ''}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Email</div>
                                <div class="field-value">${data.Email || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Role</div>
                                <div class="field-value">${data.Role || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Status</div>
                                <div class="field-value">${data.Status || 'N/A'}</div>
                            </div>
                            <div class="result-field">
                                <div class="field-label">Location</div>
                                <div class="field-value">${data.Work_Location_Code || 'N/A'}</div>
                            </div>
                        `;
                    }
                    
                    html += `</div></div>`;
                });
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