"""
EVE Online BPC Valuation Tool - Main Entry Point

This module provides the command-line interface for determining fair market value
of Blueprint Copies using Adam4EVE historical pricing data.
"""

import argparse
import sys
import csv
from typing import Optional, List, Dict
from pathlib import Path

from src.api.adam4eve_historical_client import Adam4EveHistoricalClient
from src.valuators.bpc_valuator import BPCValuator
from src.parsers.csv_parser import BPCCSVParser, validate_bpc_data
from src.models.bpc import BPC, BPCType, BPCEfficiency


def main():
    """Main entry point for the BPC valuation tool."""
    parser = argparse.ArgumentParser(
        description="Determine fair market value of EVE Online Blueprint Copies using Adam4EVE historical data",
        epilog="""
Examples:
  python main.py --create-template                           # Create blank CSV template
  python main.py --create-sample                             # Create sample CSV with examples  
  python main.py --test-api "Nyx Blueprint"                  # Test API with specific BPC
  python main.py --test-api "Nyx Blueprint" --bpc-efficiency "10/20/1"  # Test with efficiency matching
  python main.py --csv my_bpcs.csv                           # Analyze BPCs from CSV file
  python main.py --csv my_bpcs.csv --output-csv results.csv  # Generate comprehensive CSV output
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options - either single BPC or CSV file (but not required for special commands)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--bpc", 
        type=str,
        help="Single BPC name to analyze (e.g., 'Raven Blueprint Copy')"
    )
    input_group.add_argument(
        "--csv",
        type=str,
        help="CSV file path containing multiple BPCs to analyze"
    )
    
    # BPC attributes (for single BPC analysis)
    parser.add_argument(
        "--me",
        type=int,
        default=0,
        help="Material Efficiency level (0-10, default: 0)"
    )
    
    parser.add_argument(
        "--te",
        type=int,
        default=0,
        help="Time Efficiency level (0-20, default: 0)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs remaining (default: 1)"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=['ship', 'module', 'ammunition', 'structure', 'rig', 'other'],
        default='other',
        help="BPC type (default: other)"
    )
    
    # Analysis options
    parser.add_argument(
        "--region",
        type=str,
        default="The Forge",
        help="Market region for pricing analysis (default: The Forge)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to analyze (default: 30)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=['median', 'mean', 'weighted_median', 'exponential_weighted', 'conservative'],
        default='exponential_weighted',
        help="Valuation method (default: exponential_weighted)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path for batch results"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Generate comprehensive CSV output file with all items (including failed ones)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis including price statistics"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample CSV file with example data and exit"
    )
    
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a blank CSV template for user input and exit"
    )
    
    parser.add_argument(
        "--test-api",
        type=str,
        help="Test API with a specific BPC name (e.g., \"'Chivalry' Large Remote Capacitor Transmitter Blueprint\")"
    )
    
    parser.add_argument(
        "--compare-price",
        type=float,
        help="When used with --test-api, compare this price to the exponential weighted average"
    )
    
    parser.add_argument(
        "--bpc-efficiency",
        type=str,
        help="BPC efficiency in ME/TE/Runs format (e.g., '10/20/1') to find closest matching column"
    )
    
    args = parser.parse_args()
    
    # Handle sample CSV creation
    if args.create_sample:
        parser_obj = BPCCSVParser()
        parser_obj.create_sample_csv("sample_blueprints.csv")
        return
    
    # Handle blank template creation
    if args.create_template:
        parser_obj = BPCCSVParser()
        parser_obj.create_blank_template("blank_bpc_template.csv")
        return
    
    # Handle API testing
    if args.test_api:
        test_api_integration(args.test_api, args.compare_price, args.bpc_efficiency)
        return
    
    # Validate that either BPC or CSV is provided for normal operations
    if not args.bpc and not args.csv:
        parser.error("Must specify either --bpc or --csv for analysis operations")
    
    try:
        # Initialize API client and valuator with threading support
        api_client = Adam4EveHistoricalClient()
        
        # Configure threading based on input size for CSV operations
        if args.csv:
            # For CSV operations, use more threads but respect rate limits
            max_workers = min(8, max(2, len(sys.argv)))  # 2-8 threads based on complexity
            requests_per_second = 3.0  # Slightly higher for batch operations
        else:
            # For single BPC operations, keep it simple
            max_workers = 2
            requests_per_second = 2.0
            
        valuator = BPCValuator(api_client, max_workers=max_workers, requests_per_second=requests_per_second)
        
        if args.bpc:
            # Single BPC analysis
            result = analyze_single_bpc(
                valuator=valuator,
                bpc_name=args.bpc,
                me_level=args.me,
                te_level=args.te,
                runs=args.runs,
                bpc_type=args.type,
                region=args.region,
                days_back=args.days,
                valuation_method=args.method,
                detailed=args.detailed
            )
            
            if result:
                print_single_result(result, args.detailed)
        
        elif args.csv:
            # Batch CSV analysis
            results, failed_items = analyze_csv_file(
                valuator=valuator,
                csv_path=args.csv,
                region=args.region,
                days_back=args.days,
                valuation_method=args.method
            )
            
            if results or failed_items:
                print_batch_results(results, args.detailed)
                
                # Save to output file if specified
                if args.output:
                    save_results_to_csv(results, args.output)
                    print(f"\nResults saved to: {args.output}")
                
                # Save comprehensive CSV if specified
                if args.output_csv:
                    save_comprehensive_csv(results, failed_items, args.output_csv)
                    print(f"\nComprehensive results saved to: {args.output_csv}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_single_bpc(
    valuator: BPCValuator,
    bpc_name: str,
    me_level: int,
    te_level: int,
    runs: int,
    bpc_type: str,
    region: str,
    days_back: int,
    valuation_method: str,
    detailed: bool
):
    """Analyze a single BPC."""
    # Convert string type to enum
    type_mapping = {
        'ship': BPCType.SHIP,
        'module': BPCType.MODULE,
        'ammunition': BPCType.AMMUNITION,
        'structure': BPCType.STRUCTURE,
        'rig': BPCType.RIG,
        'other': BPCType.OTHER
    }
    
    bpc = BPC(
        name=bpc_name,
        item_id=0,  # Will be resolved by API client
        me_level=me_level,
        te_level=te_level,
        runs=runs,
        bpc_type=type_mapping.get(bpc_type, BPCType.OTHER)
    )
    
    return valuator.value_bpc(bpc, region, days_back, valuation_method)


def analyze_csv_file(
    valuator: BPCValuator,
    csv_path: str,
    region: str,
    days_back: int,
    valuation_method: str
) -> tuple:
    """Analyze BPCs from CSV file. Returns (successful_results, failed_items)."""
    parser = BPCCSVParser()
    
    # Parse CSV file
    bpcs = parser.parse_csv_file(csv_path)
    
    if not bpcs:
        raise ValueError("No valid BPCs found in CSV file")
    
    # Validate data
    validation = validate_bpc_data(bpcs)
    print(f"Loaded {validation['count']} BPCs from CSV")
    
    # Perform batch valuation with failure tracking
    results, failed_items = batch_value_with_failures(valuator, bpcs, region, days_back, valuation_method)
    
    return results, failed_items


def print_single_result(valuation, detailed: bool):
    """Print results for single BPC analysis."""
    print(f"\n=== BPC Valuation Analysis ===")
    print(f"BPC: {valuation.bpc.name}")
    print(f"Efficiency: ME {valuation.bpc.me_level} / TE {valuation.bpc.te_level} ({valuation.bpc.efficiency_rating})")
    print(f"Runs: {valuation.bpc.runs}")
    print(f"Region: {valuation.region}")
    print(f"Analysis Date: {valuation.analysis_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"")
    print(f"Fair Market Value: {valuation.fair_market_value:,.2f} ISK")
    print(f"Valuation Method: {valuation.valuation_method}")
    print(f"Confidence Level: {valuation.confidence_level}")
    
    if detailed:
        stats = valuation.price_statistics
        print(f"\n=== Price Statistics ===")
        print(f"Sample Size: {stats.sample_size} data points")
        print(f"Mean Price: {stats.mean_price:,.2f} ISK")
        print(f"Median Price: {stats.median_price:,.2f} ISK")
        print(f"Price Range: {stats.min_price:,.2f} - {stats.max_price:,.2f} ISK")
        print(f"Standard Deviation: {stats.std_deviation:,.2f} ISK")
        print(f"Price Trend: {stats.price_trend}")
        print(f"Confidence Score: {stats.confidence_score:.1f}/100")


def print_batch_results(valuations: List, detailed: bool, failed_items: Optional[List] = None):
    """Print results for batch analysis."""
    successful_count = len(valuations)
    failed_count = len(failed_items) if failed_items else 0
    total_count = successful_count + failed_count
    
    print(f"\n=== Batch BPC Valuation Results ===")
    print(f"Processed {total_count} BPCs: {successful_count} successful, {failed_count} failed")
    
    if successful_count > 0:
        # Calculate total values considering quantities
        individual_value = sum(v.fair_market_value for v in valuations)
        total_portfolio_value = sum(v.fair_market_value * (v.bpc.quantity if hasattr(v.bpc, 'quantity') else 1) for v in valuations)
        
        # Sort by individual fair market value descending
        sorted_valuations = sorted(valuations, key=lambda x: x.fair_market_value, reverse=True)
        
        print(f"\nIndividual BPC Value: {individual_value:,.2f} ISK")
        print(f"Total Portfolio Value: {total_portfolio_value:,.2f} ISK")
        print(f"Average BPC Value: {individual_value / len(valuations):,.2f} ISK")
        print(f"\nTop 10 Most Valuable BPCs:")
        print(f"{'Rank':<4} {'BPC Name':<35} {'Unit Value':<12} {'Qty':<3} {'Total Value':<12} {'Confidence':<10} {'Efficiency'}")
        print("-" * 95)
        
        for i, valuation in enumerate(sorted_valuations[:10], 1):
            quantity = valuation.bpc.quantity if hasattr(valuation.bpc, 'quantity') else 1
            total_value = valuation.fair_market_value * quantity
            efficiency = f"ME{valuation.bpc.me_level}/TE{valuation.bpc.te_level}"
            
            print(f"{i:<4} {valuation.bpc.name[:33]:<35} {valuation.fair_market_value:>10,.0f} {quantity:>3} {total_value:>10,.0f} {valuation.confidence_level:<10} {efficiency}")
        
        if detailed and len(sorted_valuations) > 10:
            print(f"\n... and {len(sorted_valuations) - 10} more BPCs")
    
    if failed_count > 0:
        print(f"\nFailed to value {failed_count} BPCs:")
        for failed_item in (failed_items or []):
            quantity = failed_item['bpc'].quantity if hasattr(failed_item['bpc'], 'quantity') else 1
            print(f"  ❌ {failed_item['bpc'].name} (qty: {quantity}): {failed_item['error']}")


def deduplicate_and_aggregate_bpcs(bpcs: List[BPC]) -> tuple[List[BPC], Dict[str, int]]:
    """
    Deduplicate BPCs and aggregate quantities.
    
    Args:
        bpcs: List of BPC objects (may have duplicates)
        
    Returns:
        Tuple of (unique_bpcs, quantity_map) where quantity_map maps BPC key to total quantity
    """
    unique_bpcs = {}
    quantity_map = {}
    
    for bpc in bpcs:
        # Create unique key based on BPC attributes
        key = f"{bpc.name}|{bpc.me_level}|{bpc.te_level}|{bpc.runs}"
        
        if key not in unique_bpcs:
            unique_bpcs[key] = bpc
            quantity_map[key] = bpc.quantity
        else:
            # Aggregate quantities for duplicates
            quantity_map[key] += bpc.quantity
    
    return list(unique_bpcs.values()), quantity_map


def batch_value_with_failures(valuator, bpcs, region, days_back, valuation_method):
    """Perform batch valuation while tracking failed items and using caching for duplicates."""
    # Deduplicate BPCs and get quantity mapping
    unique_bpcs, quantity_map = deduplicate_and_aggregate_bpcs(bpcs)
    
    print(f"Processing {len(bpcs)} total BPCs ({len(unique_bpcs)} unique)")
    
    # Use the multithreaded batch valuator
    try:
        valuations = valuator.batch_value_bpcs(unique_bpcs, region, days_back=days_back, valuation_method=valuation_method)
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return [], [{'bpc': bpc, 'error': str(e)} for bpc in bpcs]
    
    # Expand results based on quantities
    successful_results = []
    failed_items = []
    
    # Create a map of successful valuations by BPC key
    valuation_map = {}
    for valuation in valuations:
        key = f"{valuation.bpc.name}|{valuation.bpc.me_level}|{valuation.bpc.te_level}|{valuation.bpc.runs}"
        valuation_map[key] = valuation
    
    # Expand results based on quantities
    for bpc in unique_bpcs:
        key = f"{bpc.name}|{bpc.me_level}|{bpc.te_level}|{bpc.runs}"
        quantity = quantity_map[key]
        
        if key in valuation_map:
            # Success - replicate for each quantity
            for _ in range(quantity):
                successful_results.append(valuation_map[key])
        else:
            # Failed - create failed entries for each quantity
            for _ in range(quantity):
                failed_items.append({
                    'bpc': bpc,
                    'error': 'BPC valuation failed during batch processing'
                })
    
    return successful_results, failed_items


def save_comprehensive_csv(successful_results: List, failed_items: List, output_path: str):
    """Save comprehensive CSV with all items including failures."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'rank', 'bpc_name', 'me_level', 'te_level', 'runs', 'quantity', 'efficiency_rating',
            'fair_market_value', 'total_value', 'valuation_method', 'confidence_level', 'confidence_score',
            'sample_size', 'mean_price', 'median_price', 'price_trend', 
            'region', 'analysis_date', 'status', 'error_message'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort successful results by individual value for ranking
        sorted_results = sorted(successful_results, key=lambda x: x.fair_market_value, reverse=True)
        
        # Write successful results
        for rank, valuation in enumerate(sorted_results, 1):
            quantity = valuation.bpc.quantity if hasattr(valuation.bpc, 'quantity') else 1
            total_value = valuation.fair_market_value * quantity
            
            row = {
                'rank': rank,
                'bpc_name': valuation.bpc.name,
                'me_level': valuation.bpc.me_level,
                'te_level': valuation.bpc.te_level,
                'runs': valuation.bpc.runs,
                'quantity': quantity,
                'efficiency_rating': valuation.bpc.efficiency_rating,
                'fair_market_value': f"{valuation.fair_market_value:.2f}",
                'total_value': f"{total_value:.2f}",
                'valuation_method': valuation.valuation_method,
                'confidence_level': valuation.confidence_level,
                'confidence_score': f"{valuation.price_statistics.confidence_score:.1f}",
                'sample_size': valuation.price_statistics.sample_size,
                'mean_price': f"{valuation.price_statistics.mean_price:.2f}",
                'median_price': f"{valuation.price_statistics.median_price:.2f}",
                'price_trend': valuation.price_statistics.price_trend,
                'region': valuation.region,
                'analysis_date': valuation.analysis_date.isoformat(),
                'status': 'SUCCESS',
                'error_message': ''
            }
            writer.writerow(row)
        
        # Write failed items
        for failed_item in failed_items:
            bpc = failed_item['bpc']
            quantity = bpc.quantity if hasattr(bpc, 'quantity') else 1
            
            row = {
                'rank': '',
                'bpc_name': bpc.name,
                'me_level': bpc.me_level,
                'te_level': bpc.te_level,
                'runs': bpc.runs,
                'quantity': quantity,
                'efficiency_rating': bpc.efficiency_rating,
                'fair_market_value': '',
                'total_value': '',
                'valuation_method': '',
                'confidence_level': '',
                'confidence_score': '',
                'sample_size': '',
                'mean_price': '',
                'median_price': '',
                'price_trend': '',
                'region': '',
                'analysis_date': '',
                'status': 'FAILED',
                'error_message': failed_item['error']
            }
            writer.writerow(row)


def save_results_to_csv(valuations: List, output_path: str):
    """Save valuation results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'bpc_name', 'me_level', 'te_level', 'runs', 'efficiency_rating',
            'fair_market_value', 'valuation_method', 'confidence_level',
            'mean_price', 'median_price', 'price_trend', 'sample_size',
            'region', 'analysis_date'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for valuation in valuations:
            row = {
                'bpc_name': valuation.bpc.name,
                'me_level': valuation.bpc.me_level,
                'te_level': valuation.bpc.te_level,
                'runs': valuation.bpc.runs,
                'efficiency_rating': valuation.bpc.efficiency_rating,
                'fair_market_value': valuation.fair_market_value,
                'valuation_method': valuation.valuation_method,
                'confidence_level': valuation.confidence_level,
                'mean_price': valuation.price_statistics.mean_price,
                'median_price': valuation.price_statistics.median_price,
                'price_trend': valuation.price_statistics.price_trend,
                'sample_size': valuation.price_statistics.sample_size,
                'region': valuation.region,
                'analysis_date': valuation.analysis_date.isoformat()
            }
            writer.writerow(row)


def test_api_integration(bpc_name: str, compare_price: Optional[float] = None, bpc_efficiency: Optional[str] = None):
    """Test the API integration with a specific BPC name."""
    print(f"Testing API integration with: {bpc_name}")
    
    try:
        # Initialize API client
        api_client = Adam4EveHistoricalClient()
        
        # Step 1: Search for item ID
        print(f"\n1. Searching for item ID...")
        item_id = api_client.search_bpc_by_name(bpc_name)
        
        if item_id:
            print(f"   Found item ID: {item_id}")
        else:
            print(f"   ❌ Could not find item ID for: {bpc_name}")
            return
        
        # Step 2: Fetch historical price data
        print(f"\n2. Fetching historical price data...")
        
        # Check if BPC efficiency matching is requested
        if bpc_efficiency:
            try:
                target_efficiency = BPCEfficiency.parse(bpc_efficiency)
                print(f"   Using BPC efficiency matching for: {target_efficiency}")
                
                historical_data, matched_efficiency = api_client.get_historical_prices_with_efficiency(
                    item_id, target_efficiency, "The Forge", 30
                )
                
                if matched_efficiency:
                    print(f"   Using data from efficiency column: {matched_efficiency}")
                else:
                    print(f"   ❌ Could not find matching efficiency column")
                    return
                    
            except ValueError as e:
                print(f"   ❌ Invalid BPC efficiency format: {e}")
                return
        else:
            # Use default pricing (first available column)
            historical_data = api_client.get_historical_prices(item_id, "The Forge", 30)
        
        if historical_data:
            print(f"   Found {len(historical_data)} price data points")
            print(f"   Date range: {historical_data[-1].date.strftime('%Y-%m-%d')} to {historical_data[0].date.strftime('%Y-%m-%d')}")
            
            # Show some sample data
            print(f"\n   Recent prices:")
            for i, price_data in enumerate(historical_data[:5]):
                print(f"   {price_data.date.strftime('%Y-%m-%d')}: {price_data.price:,.0f} ISK (Volume: {price_data.volume})")
            
            # Basic statistics
            prices = [p.price for p in historical_data if p.price > 0]
            if prices:
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                
                print(f"\n   Price Statistics:")
                print(f"   Average: {avg_price:,.0f} ISK")
                print(f"   Min: {min_price:,.0f} ISK")
                print(f"   Max: {max_price:,.0f} ISK")
                
                # Calculate exponential weighted average
                from src.valuators.bpc_valuator import BPCValuator
                valuator = BPCValuator(api_client)
                
                ewa_10_day = valuator._exponential_weighted_average(historical_data, half_life_days=10)
                ewa_5_day = valuator._exponential_weighted_average(historical_data, half_life_days=5)
                
                print(f"\n   Exponential Weighted Averages:")
                print(f"   10-day half-life: {ewa_10_day:,.0f} ISK")
                print(f"   5-day half-life: {ewa_5_day:,.0f} ISK")
                
                # If compare price is provided, analyze deviation
                if compare_price:
                    print(f"\n3. Price Deviation Analysis:")
                    print(f"   Comparing price: {compare_price:,.0f} ISK")
                    
                    deviation_10 = valuator.calculate_price_deviation(historical_data, compare_price, 10)
                    deviation_5 = valuator.calculate_price_deviation(historical_data, compare_price, 5)
                    
                    print(f"\n   vs 10-day EWA:")
                    print(f"   Deviation: {deviation_10['deviation_isk']:+,.0f} ISK ({deviation_10['deviation_percent']:+.1f}%)")
                    if deviation_10['is_overpriced']:
                        print(f"   🔴 Potentially OVERPRICED (>20% above trend)")
                    elif deviation_10['is_underpriced']:
                        print(f"   🟢 Potentially UNDERPRICED (>20% below trend)")
                    else:
                        print(f"   ⚪ Within normal range")
                    
                    print(f"\n   vs 5-day EWA:")
                    print(f"   Deviation: {deviation_5['deviation_isk']:+,.0f} ISK ({deviation_5['deviation_percent']:+.1f}%)")
                    if deviation_5['is_overpriced']:
                        print(f"   🔴 Potentially OVERPRICED (>20% above recent trend)")
                    elif deviation_5['is_underpriced']:
                        print(f"   🟢 Potentially UNDERPRICED (>20% below recent trend)")
                    else:
                        print(f"   ⚪ Within normal range")
                
        else:
            print(f"   ❌ No historical price data found")
        
    except Exception as e:
        print(f"   ❌ API test failed: {e}")


if __name__ == "__main__":
    main()