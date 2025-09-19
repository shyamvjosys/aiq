# Josys Data Analysis with AI-Powered NL2SQL

This project provides **AI-powered Natural Language to SQL (NL2SQL)** querying for Josys organizational data with comprehensive analytics and insights. It uses OpenAI GPT integration to convert natural language questions into SQL queries and provides detailed analysis with cross-reference capabilities.

## ✨ Key Features

- **🧠 AI Insights & Analytics**: Intelligent analysis of query results with contextual insights
- **📊 Breakdown Analysis**: Detailed component analysis showing individual counts and intersections
- **🔍 Key Findings**: Important discoveries and patterns from your queries
- **🔗 Cross-Reference Results**: Detailed user listings with status indicators and blocker identification
- **💡 Smart Suggestions**: Interactive query suggestions based on your search
- **📋 Comprehensive Results**: Clean, readable list format with detailed user information
- **🔍 Full-Text Search**: Advanced search capabilities across all data tables

## 📊 Database Coverage

- **Devices Database**: 530+ records with asset management data (laptops, computers, phones, etc.)
- **Provisions Database**: 900+ records with user access and application provisioning data  
- **App Portfolio Database**: 4,000+ records with detailed application access, roles, and cost information

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd aiq
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key
Create a `.env` file in the project root:
```bash
echo 'OPENAI_API_KEY=your_openai_key_here' > .env
```

### 4. Start the Service
```bash
python3 start_nlp_service.py
```

The service will start and be available at `http://localhost:5000`

## 📖 Usage

### Starting the Service
```bash
python3 start_nlp_service.py
```

### Accessing the Enhanced Interface
1. Open your browser and navigate to `http://localhost:5000`
2. Ask questions in natural language
3. Get comprehensive analysis with multiple insight sections

### 🎯 Enhanced Query Experience

The interface provides **8 comprehensive sections** for every query:

1. **🧠 AI Insights & Analytics** - Intelligent analysis and contextual insights
2. **📊 Breakdown Analysis** - Individual component counts and intersections
3. **🔍 Key Findings** - Important patterns and discoveries
4. **🔗 Cross-Reference Results** - Detailed user listings with status indicators
5. **📋 Results** - Clean, readable data with detailed information
6. **💡 Smart Suggestions** - Interactive follow-up query recommendations
7. **📋 Comprehensive Summary** - Overall analysis overview
8. **📝 Generated SQL Query** - Technical details for developers

### Example Queries

**Complex Multi-Criteria Queries:**
- `"Anyone with Lenovo laptop with AWS Admin Access and also active licence of Notion app?"`
- `"List of macbook laptops assigned to employees in India"`
- `"AWS admins in Japan with GitHub access"`

**Access Analysis Queries:**
- `"Share AWS Admin usernames"`
- `"Who has GitHub access?"`
- `"List users with both Slack and AWS access"`

**Hardware Analysis Queries:**
- `"Show available MacBooks in Bangalore"`
- `"List devices owned by specific users"`
- `"How many devices are assigned in each city?"`

## 🔧 Additional Instructions

### Generating the Database
If you need to regenerate the database from the CSV files:

```bash
python3 csv_to_sqlite.py
```

This script will:
1. Read `josys-devices.csv`, `josys-provisions.csv`, and `josys-app-portfolio.csv`
2. Clean and normalize column names
3. Create `josys_data.db` SQLite database
4. Set up full-text search indexes
5. Create useful views for common queries

**Note**: The database `josys_data.db` is already included in the repository, so this step is only needed if you want to update the data with new CSV files.

### Database Schema
- **Devices Table**: Asset management with Device_Type, Manufacturer, Assigned_User_s_Email, City, Region, etc.
- **Provisions Table**: User access to 57+ applications (GitHub, Slack, AWS, Google Workspace, Notion, etc.)
- **App Portfolio Table**: Detailed application access with App, Role_s, Monthly_Expense, Account_Status, etc.

## 🛠️ Troubleshooting

### Quick Solutions

#### Database Issues
- **Database not found**: Ensure you're in the correct directory or run `python3 csv_to_sqlite.py`
- **CSV files missing**: Ensure `josys-devices.csv`, `josys-provisions.csv`, and `josys-app-portfolio.csv` are present

#### API Configuration
- **OpenAI API key missing**: Create `.env` file with `OPENAI_API_KEY=your_key_here`
- **API errors**: Verify your OpenAI API key is valid and has available credits

#### Service Issues  
- **Port 5000 in use**: Stop other services or modify port in `nlp_openai_interface.py`
- **Module errors**: Run `pip install -r requirements.txt`

#### Query Optimization
- **Better results**: Be specific, use exact terms, try rephrasing queries
- **Complex queries**: Use multi-criteria queries for comprehensive cross-reference analysis