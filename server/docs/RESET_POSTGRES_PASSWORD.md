# How to Reset PostgreSQL Password on Windows

## Option 1: Reset via Command Line (Easiest)

### Step 1: Find PostgreSQL Installation

Check where PostgreSQL is installed:

```powershell
Get-Service postgresql* | Select-Object Name, DisplayName
```

### Step 2: Reset Password using psql

**Method A - If you can connect without password:**

```powershell
# Try connecting (might work if trust authentication is enabled)
psql -U postgres

# Once connected, reset password:
ALTER USER postgres PASSWORD 'your_new_password';

# Exit
\q
```

**Method B - Using pg_hba.conf (if locked out):**

1. **Find pg_hba.conf file:**

   ```powershell
   # Usually located at:
   C:\Program Files\PostgreSQL\15\data\pg_hba.conf
   # or
   C:\Program Files\PostgreSQL\14\data\pg_hba.conf
   ```

2. **Edit pg_hba.conf as Administrator:**

   - Open file in Notepad as Administrator
   - Find the line:
     ```
     host    all             all             127.0.0.1/32            scram-sha-256
     ```
   - Change to:
     ```
     host    all             all             127.0.0.1/32            trust
     ```
   - Save the file

3. **Restart PostgreSQL Service:**

   ```powershell
   # Find the service name
   Get-Service postgresql*

   # Restart (replace with your service name)
   Restart-Service postgresql-x64-15
   ```

4. **Connect and reset password:**

   ```powershell
   psql -U postgres
   ALTER USER postgres PASSWORD 'new_password_here';
   \q
   ```

5. **Restore pg_hba.conf:**
   - Change `trust` back to `scram-sha-256`
   - Restart PostgreSQL service again

## Option 2: Reset via pgAdmin 4

1. Open pgAdmin 4
2. Right-click on "PostgreSQL 15" (or your version) → "Disconnect Server"
3. Right-click again → "Properties"
4. Click "Connect now?"
5. If it asks for password, try common defaults: `postgres`, `admin`, or blank
6. Once connected, right-click "postgres" user → "Properties"
7. Go to "Definition" tab
8. Enter new password
9. Click "Save"

## Option 3: Reinstall PostgreSQL (Last Resort)

1. Uninstall PostgreSQL completely
2. Delete data directory (backup first if needed):
   - `C:\Program Files\PostgreSQL\15\data`
3. Reinstall PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
4. Set a new password during installation

## After Resetting Password

Update your `.env` file in the server directory:

```env
DATABASE_URL=postgresql://postgres:your_new_password@localhost:5432/docuforge
```

**Important:** Replace `your_new_password` with your actual password!

## Common Default Passwords to Try

Before resetting, try these common defaults:

- `postgres`
- `admin`
- `password`
- `root`
- (blank - just press Enter)

## Test Connection

After updating your password:

```powershell
# Test with psql
psql -U postgres -d docuforge

# Test with Python script
cd server
python test_database.py
```

## Quick Commands Reference

```powershell
# Check PostgreSQL service status
Get-Service postgresql*

# Start PostgreSQL
Start-Service postgresql-x64-15

# Restart PostgreSQL
Restart-Service postgresql-x64-15

# Stop PostgreSQL
Stop-Service postgresql-x64-15

# Connect to database
psql -U postgres

# Connect to specific database
psql -U postgres -d docuforge

# List databases
psql -U postgres -c "\l"
```

## Troubleshooting

### "psql: command not found"

Add PostgreSQL to PATH:

1. Search for "Environment Variables" in Windows
2. Edit "Path" variable
3. Add: `C:\Program Files\PostgreSQL\15\bin`
4. Restart PowerShell

### "peer authentication failed"

Edit `pg_hba.conf` and change authentication method from `peer` to `md5` or `scram-sha-256`

### Still Can't Connect?

Create a new PostgreSQL user:

```powershell
# As administrator
psql -U postgres
CREATE USER docuforge_user WITH PASSWORD 'your_password';
CREATE DATABASE docuforge OWNER docuforge_user;
GRANT ALL PRIVILEGES ON DATABASE docuforge TO docuforge_user;
\q
```

Update `.env`:

```env
DATABASE_URL=postgresql://docuforge_user:your_password@localhost:5432/docuforge
```

## Need More Help?

Check PostgreSQL logs:

```powershell
# Usually located at:
C:\Program Files\PostgreSQL\15\data\log\
```
