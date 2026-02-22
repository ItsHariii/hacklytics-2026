#!/bin/bash
# Run RespiLens initial schema migration against Supabase
#
# Option 1 - Full connection string (from Supabase Dashboard > Connect > URI):
#   DATABASE_URL="postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres" ./scripts/run-supabase-migration.sh
#
# Option 2 - Password only (uses direct connection for project hfcvyckuibyyshviwxyo):
#   SUPABASE_DB_PASSWORD="your-db-password" ./scripts/run-supabase-migration.sh

set -e
MIGRATION_FILE="$(dirname "$0")/../backend/supabase/migrations/001_initial_schema.sql"

if [ -n "$DATABASE_URL" ]; then
  echo "Running migration with DATABASE_URL..."
  psql "$DATABASE_URL" -f "$MIGRATION_FILE"
elif [ -n "$SUPABASE_DB_PASSWORD" ]; then
  echo "Running migration with direct connection..."
  psql "postgresql://postgres:${SUPABASE_DB_PASSWORD}@db.hfcvyckuibyyshviwxyo.supabase.co:5432/postgres?sslmode=require" -f "$MIGRATION_FILE"
else
  echo "Error: Set DATABASE_URL or SUPABASE_DB_PASSWORD"
  echo ""
  echo "Get your connection string from: https://supabase.com/dashboard/project/hfcvyckuibyyshviwxyo/settings/database"
  echo "Or run: SUPABASE_DB_PASSWORD='your-password' $0"
  exit 1
fi

echo "Migration complete."
