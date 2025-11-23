import argparse
import logging
from pathlib import Path

from src.api.everef_contract_snapshot import EverefContractSnapshot


def main():
    parser = argparse.ArgumentParser(description="Quick Everef snapshot test client")
    parser.add_argument("--type-id", type=int, required=True, help="Type ID of the blueprint copy")
    parser.add_argument("--region-id", type=int, default=10000002, help="Region ID (default: The Forge=10000002)")
    parser.add_argument("--me", type=int, help="Material Efficiency filter")
    parser.add_argument("--te", type=int, help="Time Efficiency filter")
    parser.add_argument("--runs", type=int, help="Runs filter")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/everef_public_contracts"), help="Cache directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(filename)s:%(funcName)s: %(message)s"
    )

    client = EverefContractSnapshot(cache_dir=args.cache_dir)
    prices = client.get_prices(
        type_id=args.type_id,
        region_id=args.region_id,
        me_level=args.me,
        te_level=args.te,
        runs=args.runs,
    )

    print(f"Found {len(prices)} price points")
    for p in prices[:10]:
        print(f"{p.date.isoformat()} price={p.price:,.0f} volume={p.volume}")


if __name__ == "__main__":
    main()
