#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until PGPASSWORD=$LIQUIBASE_DATABASE_PASSWORD psql -h db -U $LIQUIBASE_DATABASE_USERNAME -d feature_engineering -c '\q'; do
  >&2 echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

>&2 echo "PostgreSQL is up - executing command"

# Execute Liquibase command
exec /opt/liquibase/liquibase \
  --changelog-file=$LIQUIBASE_CHANGELOG_FILE \
  --url=$LIQUIBASE_DATABASE_URL \
  --username=$LIQUIBASE_DATABASE_USERNAME \
  --password=$LIQUIBASE_DATABASE_PASSWORD \
  "$@" 