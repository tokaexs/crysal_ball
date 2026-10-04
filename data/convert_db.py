
from pathlib import Path
import duckdb
import pandas as pd

# Project data directory
DATA_DIR = Path(__file__).resolve().parent
CSV_DIR = DATA_DIR / "csv_exports"

CSV_DIR.mkdir(exist_ok=True)

# 1. Convert every Parquet file to CSV
for parquet_file in DATA_DIR.glob("*.parquet"):
    csv_file = CSV_DIR / f"{parquet_file.stem}.csv"

    try:
        df = pd.read_parquet(parquet_file)
        df.to_csv(csv_file, index=False)
        print(f"✅ {parquet_file.name} -> {csv_file.name}")
    except Exception as e:
        print(f"❌ Failed to convert {parquet_file.name}: {e}")

# 2. Export every table in the DuckDB database
db_path = DATA_DIR / "warehouse.duckdb"
db_csv_dir = CSV_DIR / "duckdb_tables"
db_csv_dir.mkdir(exist_ok=True)

if not db_path.exists():
    print("❌ warehouse.duckdb not found")
else:
    try:
        con = duckdb.connect(str(db_path), read_only=True)

        tables = con.execute("SHOW TABLES").fetchall()

        if not tables:
            print("⚠️ No tables found in DuckDB.")
        else:
            for (table_name,) in tables:
                # Quote the table name safely
                escaped_name = table_name.replace('"', '""')
                df = con.execute(
                    f'SELECT * FROM "{escaped_name}"'
                ).df()

                output_file = db_csv_dir / f"{table_name}.csv"
                df.to_csv(output_file, index=False)

                print(
                    f"✅ DuckDB table {table_name} "
                    f"-> duckdb_tables/{output_file.name}"
                )

        con.close()

    except Exception as e:
        print(f"❌ DuckDB export failed: {e}")

print(f"\n🎉 CSV exports saved in: {CSV_DIR}")
