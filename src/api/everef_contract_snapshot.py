import json
import logging
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import requests

from src.models.bpc import HistoricalPrice


logger = logging.getLogger(__name__)


class EverefContractSnapshot:
    """
    Lightweight client for Everef public contract snapshots.
    Downloads and caches the latest archive, extracts needed CSVs,
    and provides HistoricalPrice rows per BPC efficiency/run/region.
    """

    SNAPSHOT_URL = "https://data.everef.net/public-contracts/public-contracts-latest.v2.tar.bz2"

    def __init__(self, cache_dir: Path = Path("cache/everef_public_contracts")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tar_path = self.cache_dir / "public-contracts-latest.v2.tar.bz2"
        self.meta_path = self.cache_dir / "meta.json"
        self.contracts_path = self.cache_dir / "contracts.csv"
        self.contract_items_path = self.cache_dir / "contract_items.csv"
        self.parquet_path = self.cache_dir / "bpc_contracts.parquet"

        self._loaded = False
        self._index: Dict[Tuple[int, int], pd.DataFrame] = {}
        self._conn = duckdb.connect(database=":memory:")

    def _is_fresh(self, max_age_minutes: int = 30) -> bool:
        try:
            meta = json.loads(self.meta_path.read_text())
            scrape_end = datetime.fromisoformat(meta["scrape_end"].replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - scrape_end
            return age < timedelta(minutes=max_age_minutes)
        except Exception as exc:
            logger.debug("Failed freshness check: %s", exc)
            return False

    def _download_snapshot(self):
        logger.info("Downloading Everef public contracts snapshot...")
        with requests.get(self.SNAPSHOT_URL, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(self.tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def _extract_needed_files(self):
        logger.info("Extracting contracts.csv, contract_items.csv, meta.json")
        with tarfile.open(self.tar_path, "r:bz2") as tar:
            for member in tar.getmembers():
                base = Path(member.name).name
                if base in {"meta.json", "contracts.csv", "contract_items.csv"}:
                    target = self.cache_dir / base
                    if member.isfile():
                        with tar.extractfile(member) as src, open(target, "wb") as dst:
                            if src:
                                dst.write(src.read())

    def ensure_snapshot(self):
        data_ready = self._is_fresh(max_age_minutes=25) and self.contracts_path.exists() and self.contract_items_path.exists()
        if not data_ready:
            # Avoid hammering: if we recently downloaded within the last 10 minutes, skip re-download
            recent_download = self.tar_path.exists() and (datetime.now().timestamp() - self.tar_path.stat().st_mtime) < 600
            if recent_download:
                logger.info("Recent snapshot download detected; reusing existing files")
            else:
                self._download_snapshot()
                self._extract_needed_files()
        else:
            logger.debug("Using cached Everef snapshot")
        # Rebuild parquet if missing or stale
        parquet_stale = (
            not self.parquet_path.exists()
            or (self.meta_path.exists() and self.parquet_path.stat().st_mtime < self.meta_path.stat().st_mtime)
        )
        if parquet_stale:
            self._build_parquet()
        else:
            logger.info("Using cached parquet at %s", self.parquet_path)

    def _build_parquet(self):
        old_count = 0
        if self.parquet_path.exists():
            try:
                path_str = str(self.parquet_path).replace("'", "''")
                query = f"SELECT COUNT(*) FROM read_parquet('{path_str}')"
                old_count = self._conn.execute(query).fetchone()[0]
            except Exception as exc:
                logger.debug("Could not read existing parquet for counts: %s", exc)

        logger.info("Building BPC parquet from Everef snapshot")
        # Use DuckDB to keep memory low and filter efficiently
        self._conn.execute("SET threads TO 4")
        # Read CSVs via DuckDB
        self._conn.execute(
            """
            CREATE OR REPLACE TABLE contracts AS
            SELECT contract_id,
                   date_issued,
                   price,
                   lower(type) AS type,
                   region_id,
                   start_location_id,
                   end_location_id
            FROM read_csv_auto(?, all_varchar=FALSE)
            """,
            [str(self.contracts_path)],
        )
        self._conn.execute(
            """
            CREATE OR REPLACE TABLE contract_items AS
            SELECT contract_id,
                   type_id,
                   quantity,
                   is_blueprint_copy,
                   is_included,
                   material_efficiency,
                   time_efficiency,
                   runs
            FROM read_csv_auto(?, all_varchar=FALSE)
            """,
            [str(self.contract_items_path)],
        )
        # Filter to BPCs, included, item_exchange, single-BPC contracts
        self._conn.execute(
            """
            CREATE OR REPLACE TABLE bpc_join AS
            WITH filtered_items AS (
                SELECT *
                FROM contract_items
                WHERE is_blueprint_copy = TRUE AND is_included = TRUE
            ),
            single_item_contracts AS (
                SELECT contract_id
                FROM filtered_items
                GROUP BY contract_id
                HAVING COUNT(DISTINCT type_id) = 1
            )
            SELECT fi.contract_id,
                   fi.type_id,
                   COALESCE(fi.quantity, 1) AS quantity,
                   COALESCE(fi.material_efficiency, 0) AS material_efficiency,
                   COALESCE(fi.time_efficiency, 0) AS time_efficiency,
                   COALESCE(fi.runs, 1) AS runs,
                   c.region_id,
                   c.date_issued,
                   c.price
            FROM filtered_items fi
            JOIN single_item_contracts s USING (contract_id)
            JOIN contracts c USING (contract_id)
            WHERE c.type = 'item_exchange' AND c.price > 0
            """
        )
        self._conn.execute(
            """
            COPY (
                SELECT contract_id,
                       type_id,
                       quantity,
                       material_efficiency,
                       time_efficiency,
                       runs,
                       region_id,
                       date_issued,
                       price
                FROM bpc_join
            ) TO ? (FORMAT PARQUET, COMPRESSION 'ZSTD')
            """,
            [str(self.parquet_path)],
        )
        new_count = self._conn.execute("SELECT COUNT(*) FROM bpc_join").fetchone()[0]
        added = max(0, new_count - old_count)
        purged = max(0, old_count - new_count)
        logger.info(
            "Parquet written to %s (rows now: %d, previous: %d, added: %d, purged: %d)",
            self.parquet_path,
            new_count,
            old_count,
            added,
            purged,
        )

    def load(self):
        if self._loaded:
            return
        self.ensure_snapshot()

        # register parquet view
        path_str = str(self.parquet_path).replace("'", "''")
        query = "CREATE OR REPLACE VIEW bpc_contracts AS SELECT * FROM read_parquet('{}')".format(path_str)
        self._conn.execute(query)
        self._loaded = True

    def get_prices(
        self,
        type_id: int,
        region_id: int,
        me_level: Optional[int] = None,
        te_level: Optional[int] = None,
        runs: Optional[int] = None,
    ) -> List[HistoricalPrice]:
        """
        Return HistoricalPrice rows for the given BPC attributes from the cached snapshot.
        """
        if not self._loaded:
            self.load()

        conditions = ["type_id = ?", "region_id = ?"]
        params: List = [int(type_id), int(region_id)]
        if me_level is not None:
            conditions.append("material_efficiency = ?")
            params.append(int(me_level))
        if te_level is not None:
            conditions.append("time_efficiency = ?")
            params.append(int(te_level))
        if runs is not None:
            conditions.append("runs = ?")
            params.append(int(runs))

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT date_issued, price, quantity, material_efficiency, time_efficiency, runs
            FROM bpc_contracts
            WHERE {where_clause}
        """
        df = self._conn.execute(query, params).fetch_df()
        if df.empty:
            return []

        df["date_issued"] = pd.to_datetime(df["date_issued"], utc=True)

        return [
            HistoricalPrice(
                date=row["date_issued"].to_pydatetime(),
                price=float(row["price"]),
                volume=int(row["quantity"]),
                region=str(region_id),
            )
            for _, row in df.iterrows()
        ]
