# Database Integration Summary

## Overview

PostgreSQL has been successfully integrated into the DocuForge backend server to store all document analysis results. Every prediction and tampering detection is automatically saved to the database for future reference and analysis.

## What's New

### Database Structure

- **Table:** `analysis_results`
- **Records stored:** Every document analysis with full details
- **Automatic saving:** All predictions are saved without requiring extra API calls

### New Files Created

```
server/
├── database/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Database connection and configuration
│   ├── models.py             # SQLAlchemy models (table definitions)
│   └── crud.py               # Database operations (Create, Read, Update, Delete)
├── .env.example              # Environment variables template
├── test_database.py          # Database setup test script
└── docs/
    └── DATABASE_SETUP.md     # Complete setup guide
```

### Updated Files

- `server/api/main.py` - Added database integration to endpoints
- `server/requirements/requirements_api.txt` - Added PostgreSQL dependencies

## Quick Start

### 1. Install PostgreSQL

**Windows:** Download from [postgresql.org](https://www.postgresql.org/download/windows/)

**macOS:**

```bash
brew install postgresql@15
brew services start postgresql@15
```

**Linux:**

```bash
sudo apt install postgresql postgresql-contrib
```

### 2. Create Database

```powershell
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE docuforge;

# Verify and exit
\l
\q
```

### 3. Configure Environment

Create `.env` file in `server` directory:

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/docuforge
```

### 4. Install Dependencies

```powershell
cd server
pip install -r requirements/requirements_api.txt
```

This installs:

- `sqlalchemy>=2.0.0` - Database ORM
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter
- `python-dotenv>=1.0.0` - Environment management

### 5. Test Setup

```powershell
cd server
python test_database.py
```

This will:

- ✅ Test database connection
- ✅ Create tables automatically
- ✅ Verify CRUD operations
- ✅ Show current statistics

### 6. Start Server

```powershell
python api/main.py
```

The server will:

- Test database connection on startup
- Create tables if they don't exist
- Start accepting requests
- Automatically save all analyses

## API Endpoints

### Existing (Enhanced)

- `POST /predict` - Now saves results to database automatically
- `POST /detect-tampering` - Enhanced with database storage

### New Endpoints

#### Get Specific Analysis

```http
GET /analysis/{id}
```

Query Parameters:

- `include_images` (bool) - Include base64 images (default: true)

Example:

```bash
curl http://localhost:8000/analysis/550e8400-e29b-41d4-a716-446655440000
```

#### Get Analysis History

```http
GET /analysis/history
```

Query Parameters:

- `skip` (int) - Records to skip (pagination)
- `limit` (int) - Max records (max 100, default 50)
- `user_id` (string) - Filter by user
- `prediction` (string) - Filter by 'authentic' or 'forged'

Example:

```bash
curl "http://localhost:8000/analysis/history?limit=10&prediction=forged"
```

#### Get Recent Analyses

```http
GET /analysis/recent
```

Query Parameters:

- `hours` (int) - Hours to look back (default 24)
- `limit` (int) - Max records (default 50)

Example:

```bash
curl "http://localhost:8000/analysis/recent?hours=48"
```

#### Get Statistics

```http
GET /statistics
```

Query Parameters:

- `user_id` (string) - Filter by user (optional)

Returns:

```json
{
  "total_analyses": 150,
  "authentic_count": 90,
  "forged_count": 60,
  "successful_count": 148,
  "failed_count": 2,
  "average_confidence": 0.8234,
  "average_probability": 0.3456,
  "average_processing_time": 2.3456
}
```

#### Delete Analysis

```http
DELETE /analysis/{id}
```

Example:

```bash
curl -X DELETE http://localhost:8000/analysis/550e8400-e29b-41d4-a716-446655440000
```

## Database Schema

### `analysis_results` Table

| Column             | Type        | Description                       |
| ------------------ | ----------- | --------------------------------- |
| `id`               | UUID        | Primary key (auto-generated)      |
| `created_at`       | DateTime    | Timestamp of analysis             |
| `filename`         | String(255) | Original filename                 |
| `file_size`        | Integer     | File size in bytes                |
| `user_id`          | String(255) | User ID (nullable)                |
| `user_email`       | String(255) | User email (nullable)             |
| `prediction`       | String(50)  | 'authentic' or 'forged'           |
| `probability`      | Float       | Probability of being forged (0-1) |
| `confidence`       | Float       | Confidence score (0-1)            |
| `heatmap`          | Text        | Base64 encoded heatmap            |
| `mask`             | Text        | Base64 encoded mask               |
| `tampered_regions` | Text        | Base64 encoded regions image      |
| `model_version`    | String(50)  | Model version used                |
| `processing_time`  | Float       | Processing time (seconds)         |
| `success`          | Boolean     | Analysis succeeded?               |
| `error_message`    | Text        | Error if failed                   |

**Indexes:**

- Primary key on `id`
- Index on `created_at` (for recent queries)
- Index on `user_id` (for user filtering)

## Usage Examples

### Python

```python
import requests

# 1. Make prediction (automatically saved)
with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Saved with ID: {result['id']}")

# 2. Retrieve saved analysis
analysis = requests.get(
    f"http://localhost:8000/analysis/{result['id']}"
).json()

# 3. Get user's analysis history
history = requests.get(
    "http://localhost:8000/analysis/history",
    params={"user_id": "auth0|123", "limit": 10}
).json()

# 4. Get statistics
stats = requests.get("http://localhost:8000/statistics").json()
print(f"Total analyses: {stats['total_analyses']}")
print(f"Forged documents: {stats['forged_count']}")

# 5. Delete old analysis
requests.delete(f"http://localhost:8000/analysis/{old_id}")
```

### JavaScript/TypeScript

```typescript
// 1. Upload and analyze
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});
const result = await response.json();
console.log("Analysis ID:", result.id);

// 2. Get history
const history = await fetch(
  "http://localhost:8000/analysis/history?limit=20"
).then((r) => r.json());

// 3. Get statistics
const stats = await fetch("http://localhost:8000/statistics").then((r) =>
  r.json()
);
```

### cURL

```bash
# Upload and analyze
curl -X POST http://localhost:8000/predict \
  -F "file=@document.jpg"

# Get analysis by ID
curl http://localhost:8000/analysis/550e8400-e29b-41d4-a716-446655440000

# Get recent analyses
curl "http://localhost:8000/analysis/recent?hours=24"

# Get statistics
curl http://localhost:8000/statistics

# Delete analysis
curl -X DELETE http://localhost:8000/analysis/550e8400-e29b-41d4-a716-446655440000
```

## Database Management

### View Records

```sql
-- Connect to database
psql -U postgres -d docuforge

-- View all analyses
SELECT id, filename, prediction, created_at
FROM analysis_results
ORDER BY created_at DESC
LIMIT 10;

-- Count by prediction type
SELECT prediction, COUNT(*)
FROM analysis_results
GROUP BY prediction;

-- Average confidence by prediction
SELECT prediction, AVG(confidence)
FROM analysis_results
GROUP BY prediction;
```

### Backup and Restore

```powershell
# Backup
pg_dump -U postgres docuforge > backup.sql

# Restore
psql -U postgres docuforge < backup.sql
```

### Clear Data

```python
# Using Python
from database import crud
from database.config import SessionLocal

db = SessionLocal()

# Delete records older than 30 days
deleted = crud.delete_old_results(db, days=30)
print(f"Deleted {deleted} old records")
```

Or SQL:

```sql
TRUNCATE TABLE analysis_results;  -- Remove all data
```

## Features

### Automatic Storage

- Every prediction is saved automatically
- No need for separate save calls
- Includes all analysis data (images, metadata)

### Query Capabilities

- Get analysis by ID
- List analyses with pagination
- Filter by user, prediction type
- Get recent analyses
- View statistics and aggregates

### Error Handling

- Graceful fallback if database is unavailable
- Server continues to work even if DB connection fails
- Errors logged for debugging

### Performance

- Connection pooling (10 connections)
- Indexed queries for fast retrieval
- Efficient pagination
- Optional image exclusion for lighter responses

### Data Management

- Automatic cleanup functions
- Delete by ID
- Delete old records by date
- Full backup/restore support

## Configuration

### Environment Variables

```env
# Required
DATABASE_URL=postgresql://user:password@host:port/database

# Optional (parsed from DATABASE_URL if not set)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=docuforge
DB_USER=postgres
DB_PASSWORD=your_password
```

### Connection Pool Settings

In `database/config.py`:

```python
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # Verify connections
    pool_size=10,            # Connection pool size
    max_overflow=20          # Extra connections allowed
)
```

## Troubleshooting

### Connection Issues

**Problem:** `could not connect to server`

**Solution:**

1. Check PostgreSQL is running
2. Verify DATABASE_URL in `.env`
3. Test connection: `psql -U postgres -d docuforge`

### Authentication Failed

**Problem:** `password authentication failed`

**Solution:**

1. Verify password in DATABASE_URL
2. Reset password: `ALTER USER postgres PASSWORD 'newpass';`

### Database Not Found

**Problem:** `database "docuforge" does not exist`

**Solution:**

```bash
psql -U postgres
CREATE DATABASE docuforge;
\q
```

### Import Errors

**Problem:** `No module named 'sqlalchemy'`

**Solution:**

```powershell
pip install -r requirements/requirements_api.txt
```

## Next Steps

### For Production

1. **Use Environment Variables** - Never commit credentials
2. **Enable SSL** - Secure database connections
3. **Regular Backups** - Automated backup schedule
4. **Monitor Performance** - Track query times
5. **Index Optimization** - Add indexes as needed
6. **Data Retention Policy** - Regularly clean old records

### Potential Enhancements

1. **User Analytics Dashboard** - Visualize statistics
2. **Export Functionality** - Export analyses to CSV/JSON
3. **Batch Operations** - Bulk upload and analysis
4. **Search Functionality** - Full-text search on filenames
5. **Webhooks** - Notify on analysis completion
6. **Rate Limiting** - Track API usage per user

## Resources

- [Full Setup Guide](./DATABASE_SETUP.md)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [FastAPI Database Tutorial](https://fastapi.tiangolo.com/tutorial/sql-databases/)

## Support

For issues or questions:

1. Check the [DATABASE_SETUP.md](./DATABASE_SETUP.md) guide
2. Run `python test_database.py` to verify setup
3. Check PostgreSQL logs for errors
4. Verify `.env` configuration

---

**Status:** ✅ Fully Integrated and Tested
**Last Updated:** October 18, 2025
