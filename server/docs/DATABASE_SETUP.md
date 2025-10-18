# PostgreSQL Database Setup Guide

This guide will help you set up PostgreSQL for the DocuForge backend server.

## Prerequisites

- PostgreSQL installed on your system
- Python environment set up with required packages

## Installation

### Windows

1. **Download PostgreSQL**

   - Visit [PostgreSQL Official Website](https://www.postgresql.org/download/windows/)
   - Download the installer for Windows
   - Run the installer and follow the setup wizard

2. **During Installation**

   - Set a password for the `postgres` superuser (remember this!)
   - Default port: 5432 (keep default unless you have a conflict)
   - Select all components (PostgreSQL Server, pgAdmin 4, Command Line Tools)

3. **Verify Installation**
   ```powershell
   # Check if PostgreSQL is running
   Get-Service postgresql*
   ```

### macOS

```bash
# Using Homebrew
brew install postgresql@15
brew services start postgresql@15
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## Database Setup

### 1. Create Database

#### Using psql (Command Line)

```powershell
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE docuforge;

# Verify
\l

# Exit
\q
```

#### Using pgAdmin 4 (GUI)

1. Open pgAdmin 4
2. Connect to your PostgreSQL server
3. Right-click on "Databases" → "Create" → "Database"
4. Name: `docuforge`
5. Click "Save"

### 2. Configure Environment Variables

Create a `.env` file in the `server` directory:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and update the DATABASE_URL:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/docuforge
```

**Connection String Format:**

```
postgresql://[username]:[password]@[host]:[port]/[database]
```

**Example configurations:**

```env
# Local development (default)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/docuforge

# Custom port
DATABASE_URL=postgresql://postgres:mypassword@localhost:5433/docuforge

# Remote database
DATABASE_URL=postgresql://user:pass@db.example.com:5432/docuforge

# With special characters in password (URL encode them)
DATABASE_URL=postgresql://user:p%40ssw0rd@localhost:5432/docuforge
```

### 3. Install Python Dependencies

```powershell
cd server
pip install -r requirements/requirements_api.txt
```

This installs:

- `sqlalchemy` - ORM for database operations
- `psycopg2-binary` - PostgreSQL adapter for Python
- `python-dotenv` - Environment variable management

### 4. Initialize Database Tables

The tables will be created automatically when you start the server for the first time. The server will:

1. Test the database connection
2. Create the `analysis_results` table if it doesn't exist
3. Start accepting requests

## Database Schema

### `analysis_results` Table

| Column             | Type     | Description                       |
| ------------------ | -------- | --------------------------------- |
| `id`               | UUID     | Primary key (auto-generated)      |
| `created_at`       | DateTime | Timestamp of analysis             |
| `filename`         | String   | Original filename                 |
| `file_size`        | Integer  | File size in bytes                |
| `user_id`          | String   | User ID (if auth enabled)         |
| `user_email`       | String   | User email (if auth enabled)      |
| `prediction`       | String   | 'authentic' or 'forged'           |
| `probability`      | Float    | Probability of being forged (0-1) |
| `confidence`       | Float    | Confidence score (0-1)            |
| `heatmap`          | Text     | Base64 encoded heatmap image      |
| `mask`             | Text     | Base64 encoded mask image         |
| `tampered_regions` | Text     | Base64 encoded regions image      |
| `model_version`    | String   | Model version used                |
| `processing_time`  | Float    | Processing time in seconds        |
| `success`          | Boolean  | Whether analysis succeeded        |
| `error_message`    | Text     | Error message if failed           |

## Testing Database Connection

### Using Python

Create a test script `test_db.py`:

```python
from database.config import test_connection, init_db

print("Testing database connection...")
if test_connection():
    print("✅ Connection successful!")
    print("Initializing tables...")
    init_db()
    print("✅ Tables created!")
else:
    print("❌ Connection failed!")
```

Run it:

```powershell
cd server
python test_db.py
```

### Using psql

```powershell
# Connect to database
psql -U postgres -d docuforge

# List tables
\dt

# View table schema
\d analysis_results

# Count records
SELECT COUNT(*) FROM analysis_results;

# View recent analyses
SELECT id, filename, prediction, created_at
FROM analysis_results
ORDER BY created_at DESC
LIMIT 10;
```

## API Endpoints

Once the database is set up, the following endpoints are available:

### Analysis Endpoints

- `POST /predict` - Analyze document (saves to DB)
- `GET /analysis/{id}` - Get specific analysis
- `GET /analysis/history` - Get analysis history
- `GET /analysis/recent` - Get recent analyses
- `GET /statistics` - Get analysis statistics
- `DELETE /analysis/{id}` - Delete analysis

### Example Usage

```python
import requests

# Make prediction (automatically saved to DB)
response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("document.jpg", "rb")}
)
result = response.json()
analysis_id = result["id"]

# Get saved analysis
analysis = requests.get(f"http://localhost:8000/analysis/{analysis_id}")

# Get history
history = requests.get("http://localhost:8000/analysis/history?limit=10")

# Get statistics
stats = requests.get("http://localhost:8000/statistics")
```

## Troubleshooting

### Connection Refused

```
ERROR: could not connect to server: Connection refused
```

**Solution:**

- Ensure PostgreSQL service is running
- Windows: Check Services (search for "postgresql")
- Verify port 5432 is not blocked

### Authentication Failed

```
ERROR: password authentication failed for user "postgres"
```

**Solution:**

- Verify password in DATABASE_URL
- Reset password if needed:
  ```powershell
  psql -U postgres
  ALTER USER postgres PASSWORD 'newpassword';
  ```

### Database Does Not Exist

```
ERROR: database "docuforge" does not exist
```

**Solution:**

- Create the database first (see step 1 above)

### Port Already in Use

```
ERROR: could not bind to port 5432
```

**Solution:**

- Check if another PostgreSQL instance is running
- Use a different port and update DATABASE_URL

### Import Errors

```
ModuleNotFoundError: No module named 'sqlalchemy'
```

**Solution:**

```powershell
pip install -r requirements/requirements_api.txt
```

## Database Management

### Backup Database

```powershell
# Backup to file
pg_dump -U postgres docuforge > backup.sql

# Restore from backup
psql -U postgres docuforge < backup.sql
```

### Clear All Data

```sql
-- Connect to database
psql -U postgres -d docuforge

-- Truncate table (keeps structure, removes data)
TRUNCATE TABLE analysis_results;

-- Or drop and recreate (removes everything)
DROP TABLE analysis_results;
```

The server will recreate the table on next startup.

### View Database Size

```sql
SELECT pg_size_pretty(pg_database_size('docuforge'));
```

### Delete Old Records

Use the API or direct SQL:

```sql
-- Delete records older than 30 days
DELETE FROM analysis_results
WHERE created_at < NOW() - INTERVAL '30 days';
```

Or use the CRUD function:

```python
from database import crud
from database.config import SessionLocal

db = SessionLocal()
deleted_count = crud.delete_old_results(db, days=30)
print(f"Deleted {deleted_count} old records")
```

## Production Deployment

For production, consider:

1. **Use Environment Variables** - Never commit credentials
2. **Enable SSL** - Use SSL connections to database
3. **Connection Pooling** - Configured by default (10 connections)
4. **Regular Backups** - Schedule automated backups
5. **Monitoring** - Monitor database size and performance
6. **Indexes** - Already added on frequently queried columns
7. **Security** - Use strong passwords, restrict access

### Example Production DATABASE_URL

```env
# With SSL
DATABASE_URL=postgresql://user:pass@db.example.com:5432/docuforge?sslmode=require

# With connection pooling parameters
DATABASE_URL=postgresql://user:pass@db.example.com:5432/docuforge?sslmode=require&pool_size=20&max_overflow=10
```

## Additional Resources

- [PostgreSQL Official Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)
- [pgAdmin 4 Documentation](https://www.pgadmin.org/docs/)
