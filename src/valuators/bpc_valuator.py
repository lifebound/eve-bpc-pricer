"""
BPC Valuator

This module provides functionality to determine fair market value of Blueprint Copies
using historical pricing data and various valuation methods.
"""

import statistics
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.api.adam4eve_historical_client import Adam4EveHistoricalClient
from src.models.bpc import BPC, BPCValuation, HistoricalPrice, PriceStatistics


class RateLimiter:
    """Thread-safe rate limiter for API requests."""
    
    def __init__(self, max_requests_per_second: float = 2.0):
        self.max_requests_per_second = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
                
            self.last_request_time = time.time()


class BPCValuator:
    """Valuator for Blueprint Copy fair market value analysis."""
    
    def __init__(self, api_client: Adam4EveHistoricalClient, max_workers: int = 4, requests_per_second: float = 2.0):
        """
        Initialize the BPC valuator.
        
        Args:
            api_client: Adam4EVE historical data client
            max_workers: Maximum number of concurrent threads
            requests_per_second: Maximum API requests per second
        """
        self.api_client = api_client
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_second)
        self.price_cache = {}  # Cache for pricing data
        
        # Thread-safe cache for valuations
        self._cache_lock = threading.Lock()
        self._valuation_cache = {}
    
    def calculate_price_statistics(self, historical_prices: List[HistoricalPrice]) -> PriceStatistics:
        """
        Calculate statistical analysis of historical prices.
        
        Args:
            historical_prices: List of historical price data
            
        Returns:
            PriceStatistics object with analysis
        """
        if not historical_prices:
            return PriceStatistics(
                mean_price=0.0,
                median_price=0.0,
                min_price=0.0,
                max_price=0.0,
                std_deviation=0.0,
                sample_size=0,
                price_trend="unknown",
                confidence_score=0.0
            )
        
        prices = [p.price for p in historical_prices if p.price > 0]
        
        if not prices:
            return PriceStatistics(
                mean_price=0.0,
                median_price=0.0,
                min_price=0.0,
                max_price=0.0,
                std_deviation=0.0,
                sample_size=0,
                price_trend="unknown",
                confidence_score=0.0
            )
        
        # Calculate basic statistics
        mean_price = statistics.mean(prices)
        median_price = statistics.median(prices)
        min_price = min(prices)
        max_price = max(prices)
        std_dev = statistics.stdev(prices) if len(prices) > 1 else 0.0
        
        # Determine price trend
        price_trend = self._analyze_price_trend(historical_prices)
        
        # Calculate confidence score based on sample size and consistency
        confidence_score = self._calculate_confidence_score(len(prices), std_dev, mean_price)
        
        return PriceStatistics(
            mean_price=mean_price,
            median_price=median_price,
            min_price=min_price,
            max_price=max_price,
            std_deviation=std_dev,
            sample_size=len(prices),
            price_trend=price_trend,
            confidence_score=confidence_score
        )
    
    def _analyze_price_trend(self, historical_prices: List[HistoricalPrice]) -> str:
        """
        Analyze price trend over time.
        
        Args:
            historical_prices: Sorted historical price data
            
        Returns:
            Trend description: "rising", "falling", "stable"
        """
        if len(historical_prices) < 3:
            return "unknown"
        
        # Sort by date
        sorted_prices = sorted(historical_prices, key=lambda x: x.date)
        
        # Compare first and last third of data
        third = len(sorted_prices) // 3
        early_avg = statistics.mean([p.price for p in sorted_prices[:third] if p.price > 0])
        late_avg = statistics.mean([p.price for p in sorted_prices[-third:] if p.price > 0])
        
        if late_avg > early_avg * 1.1:  # 10% increase
            return "rising"
        elif late_avg < early_avg * 0.9:  # 10% decrease
            return "falling"
        else:
            return "stable"
    
    def _calculate_confidence_score(self, sample_size: int, std_dev: float, mean_price: float) -> float:
        """
        Calculate confidence score for the valuation.
        
        Args:
            sample_size: Number of price data points
            std_dev: Standard deviation of prices
            mean_price: Mean price
            
        Returns:
            Confidence score from 0-100
        """
        if mean_price == 0:
            return 0.0
        
        # Base score from sample size
        size_score = min(sample_size * 2, 50)  # Max 50 points for sample size
        
        # Consistency score (lower std dev = higher confidence)
        coefficient_of_variation = std_dev / mean_price if mean_price > 0 else 1.0
        consistency_score = max(0, 50 - (coefficient_of_variation * 100))
        
        return min(size_score + consistency_score, 100)
    
    def value_bpc(
        self,
        bpc: BPC,
        region: str = "The Forge",
        days_back: int = 30,
        valuation_method: str = "exponential_weighted"
    ) -> BPCValuation:
        """
        Value a single BPC using historical market data.
        
        Args:
            bpc: BPC object to value
            region: Market region for analysis
            days_back: Number of days of historical data to analyze
            valuation_method: Method for calculating fair value
            
        Returns:
            BPCValuation object with pricing analysis
            
        Raises:
            ValueError: If BPC cannot be valued
            ConnectionError: If API request fails
        """
        return self._value_single_bpc_internal(bpc, region, days_back, valuation_method)
    
    def _value_single_bpc_threadsafe(self, bpc: BPC, region: str, days_back: int, valuation_method: str, bpc_key: str) -> tuple:
        """
        Thread-safe wrapper for valuing a single BPC.
        
        Returns:
            Tuple of (bpc_key, valuation_result_or_exception)
        """
        try:
            # Rate limit API requests
            self.rate_limiter.wait_if_needed()
            
            # Check cache first
            with self._cache_lock:
                if bpc_key in self._valuation_cache:
                    return bpc_key, self._valuation_cache[bpc_key]
            
            # Perform valuation
            result = self._value_single_bpc_internal(bpc, region, days_back, valuation_method)
            
            # Cache result
            with self._cache_lock:
                self._valuation_cache[bpc_key] = result
                
            return bpc_key, result
            
        except Exception as e:
            return bpc_key, e
    
    def _value_single_bpc_internal(
        self,
        bpc: BPC,
        region: str,
        days_back: int,
        valuation_method: str
    ) -> BPCValuation:
        """
        Calculate fair market value for a BPC.
        
        Args:
            bpc: BPC object to value
            region: Market region for analysis
            days_back: Number of days of historical data
            valuation_method: Method to use for valuation
            
        Returns:
            BPCValuation object with complete analysis
        """
        # Get item ID for the BPC
        item_id = self.api_client.search_bpc_by_name(bpc.name)
        if not item_id:
            raise ValueError(f"Could not find item ID for BPC: {bpc.name}")
        
        # Determine if we should use efficiency matching or default pricing
        use_efficiency_matching = not (bpc.me_level == 0 and bpc.te_level == 0 and bpc.runs == 1)
        
        if use_efficiency_matching:
            # Use efficiency matching to find the closest column
            from src.models.bpc import BPCEfficiency
            target_efficiency = BPCEfficiency(bpc.me_level, bpc.te_level, bpc.runs)
            historical_prices, matched_efficiency = self.api_client.get_historical_prices_with_efficiency(
                item_id, target_efficiency, region, days_back
            )
            if matched_efficiency:
                print(f"Using efficiency matching: {target_efficiency} -> {matched_efficiency}")
            else:
                print(f"No efficiency match found, falling back to default pricing")
                historical_prices = self.api_client.get_historical_prices(item_id, region, days_back)
        else:
            # Use default pricing (first available price column)
            historical_prices = self.api_client.get_historical_prices(item_id, region, days_back)
        
        # Calculate price statistics
        price_stats = self.calculate_price_statistics(historical_prices)
        
        # Determine fair market value based on method
        fair_value = self._calculate_fair_value(historical_prices, bpc, valuation_method)
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(price_stats.confidence_score)
        
        return BPCValuation(
            bpc=bpc,
            fair_market_value=fair_value,
            valuation_method=valuation_method,
            price_statistics=price_stats,
            historical_data=historical_prices,
            region=region,
            analysis_date=datetime.now(),
            confidence_level=confidence_level
        )
    
    def _calculate_fair_value(
        self,
        historical_prices: List[HistoricalPrice],
        bpc: BPC,
        method: str
    ) -> float:
        """
        Calculate fair market value using specified method.
        
        Args:
            historical_prices: Historical price data
            bpc: BPC object being valued
            method: Valuation method to use
            
        Returns:
            Fair market value in ISK
        """
        if not historical_prices:
            return 0.0
        
        prices = [p.price for p in historical_prices if p.price > 0]
        
        if not prices:
            return 0.0
        
        if method == "median":
            base_value = statistics.median(prices)
        elif method == "mean":
            base_value = statistics.mean(prices)
        elif method == "weighted_median":
            # Weight recent prices more heavily
            base_value = self._weighted_median(historical_prices)
        elif method == "exponential_weighted":
            # Exponentially-weighted average with 10-day half-life
            base_value = self._exponential_weighted_average(historical_prices, half_life_days=10)
        elif method == "conservative":
            # Use 25th percentile for conservative estimate
            sorted_prices = sorted(prices)
            percentile_25 = sorted_prices[len(sorted_prices) // 4]
            base_value = percentile_25
        else:
            # Default to exponential weighted average
            base_value = self._exponential_weighted_average(historical_prices, half_life_days=10)
        
        # Apply efficiency adjustments
        efficiency_multiplier = self._get_efficiency_multiplier(bpc)
        
        return base_value * efficiency_multiplier
    
    def _weighted_median(self, historical_prices: List[HistoricalPrice]) -> float:
        """
        Calculate weighted median giving more weight to recent prices.
        
        Args:
            historical_prices: Historical price data
            
        Returns:
            Weighted median price
        """
        if not historical_prices:
            return 0.0
        
        # Sort by date
        sorted_prices = sorted(historical_prices, key=lambda x: x.date)
        
        # Create weights (more recent = higher weight)
        weights = []
        for i, price_data in enumerate(sorted_prices):
            # Linear weight increase (most recent gets highest weight)
            weight = i + 1
            weights.extend([price_data.price] * weight)
        
        return statistics.median(weights) if weights else 0.0
        
    def _exponential_weighted_average(self, historical_prices: List[HistoricalPrice], half_life_days: float = 10.0) -> float:
        """
        Calculate exponential weighted average giving exponentially more weight to recent prices.
        
        Args:
            historical_prices: Historical price data (should be sorted by date)
            half_life_days: Number of days for the weight to decay to 50%
            
        Returns:
            Exponentially weighted average price
        """
        if not historical_prices:
            return 0.0
        
        import math
        
        # Sort by date (most recent first)
        sorted_prices = sorted(historical_prices, key=lambda x: x.date, reverse=True)
        
        if len(sorted_prices) == 1:
            return sorted_prices[0].price
        
        # Calculate decay factor from half-life
        # If half_life = 10 days, then lambda = ln(2) / 10
        decay_factor = math.log(2) / half_life_days
        
        # Most recent date (reference point)
        most_recent_date = sorted_prices[0].date
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for price_data in sorted_prices:
            if price_data.price <= 0:
                continue
                
            # Calculate days back from most recent
            days_back = (most_recent_date - price_data.date).days
            
            # Calculate exponential weight
            # weight = e^(-lambda * days_back)
            weight = math.exp(-decay_factor * days_back)
            
            weighted_sum += price_data.price * weight
            weight_sum += weight
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.0
    
    def calculate_price_deviation(self, historical_prices: List[HistoricalPrice], current_price: float, half_life_days: float = 10.0) -> Dict[str, float]:
        """
        Calculate how much a current price deviates from the exponential weighted average.
        
        Args:
            historical_prices: Historical price data
            current_price: Current asking price to compare
            half_life_days: Half-life for exponential weighting
            
        Returns:
            Dictionary with deviation analysis
        """
        if not historical_prices or current_price <= 0:
            return {
                "ewa_price": 0.0,
                "deviation_isk": 0.0,
                "deviation_percent": 0.0,
                "is_overpriced": False,
                "is_underpriced": False
            }
        
        ewa_price = self._exponential_weighted_average(historical_prices, half_life_days)
        
        if ewa_price <= 0:
            return {
                "ewa_price": 0.0,
                "deviation_isk": 0.0,
                "deviation_percent": 0.0,
                "is_overpriced": False,
                "is_underpriced": False
            }
        
        deviation_isk = current_price - ewa_price
        deviation_percent = (deviation_isk / ewa_price) * 100
        
        # Consider >20% above EWA as potentially overpriced
        # Consider >20% below EWA as potentially underpriced
        is_overpriced = deviation_percent > 20.0
        is_underpriced = deviation_percent < -20.0
        
        return {
            "ewa_price": ewa_price,
            "deviation_isk": deviation_isk,
            "deviation_percent": deviation_percent,
            "is_overpriced": is_overpriced,
            "is_underpriced": is_underpriced
        }
    
    def _get_efficiency_multiplier(self, bpc: BPC) -> float:
        """
        Get multiplier based on BPC efficiency levels.
        
        Args:
            bpc: BPC object
            
        Returns:
            Multiplier for efficiency levels
        """
        # Base efficiency scoring
        me_score = min(bpc.me_level / 10.0, 1.0)  # ME 0-10 normalized
        te_score = min(bpc.te_level / 20.0, 1.0)  # TE 0-20 normalized
        
        # Efficiency multiplier (perfect BPCs worth more)
        efficiency_avg = (me_score + te_score) / 2
        
        # Multiplier ranges from 0.8 (poor) to 1.5 (perfect)
        multiplier = 0.8 + (efficiency_avg * 0.7)
        
        # Bonus for perfect BPCs
        if bpc.me_level == 10 and bpc.te_level == 20:
            multiplier *= 1.1  # 10% bonus for perfect
        
        return multiplier
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """
        Convert confidence score to descriptive level.
        
        Args:
            confidence_score: Numerical confidence score
            
        Returns:
            Confidence level description
        """
        if confidence_score >= 80:
            return "High"
        elif confidence_score >= 50:
            return "Medium"
        else:
            return "Low"
    
    def batch_value_bpcs(
        self,
        bpcs: List[BPC],
        region: str = "The Forge",
        **kwargs
    ) -> List[BPCValuation]:
        """
        Value multiple BPCs in parallel using thread pool.
        
        Args:
            bpcs: List of BPC objects to value
            region: Market region for analysis
            **kwargs: Additional parameters for valuation
            
        Returns:
            List of BPCValuation objects
        """
        if not bpcs:
            return []
            
        days_back = kwargs.get('days_back', 30)
        valuation_method = kwargs.get('valuation_method', 'exponential_weighted')
        
        print(f"Processing {len(bpcs)} BPCs using {self.max_workers} threads (max {self.rate_limiter.max_requests_per_second} req/s)")
        
        # Create tasks for thread pool
        tasks = []
        for bpc in bpcs:
            bpc_key = f"{bpc.name}|{bpc.me_level}|{bpc.te_level}|{bpc.runs}"
            tasks.append((bpc, region, days_back, valuation_method, bpc_key))
        
        results = []
        failed_count = 0
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._value_single_bpc_threadsafe, bpc, region, days_back, valuation_method, bpc_key): (bpc, bpc_key)
                for bpc, region, days_back, valuation_method, bpc_key in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                bpc, bpc_key = future_to_task[future]
                try:
                    returned_key, result = future.result()
                    if isinstance(result, Exception):
                        print(f"Error valuing BPC {bpc.name}: {result}")
                        failed_count += 1
                    else:
                        results.append(result)
                        print(f"✓ Completed: {bpc.name} ({len(results)}/{len(bpcs) - failed_count})")
                except Exception as e:
                    print(f"Thread error for BPC {bpc.name}: {e}")
                    failed_count += 1
        
        print(f"Batch processing complete: {len(results)} successful, {failed_count} failed")
        return results