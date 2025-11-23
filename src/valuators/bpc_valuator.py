"""
BPC Valuator

This module provides functionality to determine fair market value of Blueprint Copies
using historical pricing data and various valuation methods.
"""

import logging
import statistics
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.api.adam4eve_historical_client import Adam4EveHistoricalClient
from src.api.everef_contract_snapshot import EverefContractSnapshot
from src.models.bpc import BPC, BPCValuation, HistoricalPrice, PriceStatistics

logger = logging.getLogger(__name__)


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
    
    def __init__(self, api_client: Adam4EveHistoricalClient, max_workers: int = 4, requests_per_second: float = 2.0, snapshot_client: Optional[EverefContractSnapshot] = None):
        """
        Initialize the BPC valuator.
        
        Args:
            api_client: Adam4EVE historical data client
            max_workers: Maximum number of concurrent threads
            requests_per_second: Maximum API requests per second
            snapshot_client: Optional Everef snapshot client for contract data
        """
        self.api_client = api_client
        self.snapshot_client = snapshot_client
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_second)
        self.price_cache = {}  # Cache for pricing data
        
        # Thread-safe cache for valuations
        self._cache_lock = threading.Lock()
        self._valuation_cache = {}
        self._thread_local = threading.local()
        
        # Shared caches for item IDs and contract HTML (to avoid repeated Adam4EVE calls)
        self._item_id_cache: Dict[str, int] = {}
        self._contract_html_cache: Dict[tuple, str] = {}
        self._html_cache_lock = threading.Lock()
    
    def _get_thread_client(self) -> Adam4EveHistoricalClient:
        """
        Provide a thread-confined Adam4EveHistoricalClient to avoid sharing a Session across threads.
        Reuses the caller-supplied client on the main thread and spins up per-thread clones otherwise.
        """
        client = getattr(self._thread_local, "client", None)
        if client is None:
            # Clone using the same client type and rate limit; copy headers for parity.
            base_cls = type(self.api_client)
            rate_limit = getattr(self.api_client, "rate_limit", 1.0)
            client = base_cls(rate_limit=rate_limit)
            try:
                client.session.headers.update(self.api_client.session.headers)
            except Exception:
                pass
            # Share contract page cache across thread-local clients to avoid duplicate fetches
            try:
                client._contract_page_cache = self.api_client._contract_page_cache
            except Exception:
                pass
            self._thread_local.client = client
        return client
    
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
        api_client = getattr(self._thread_local, "client", self.api_client)
        return self._value_single_bpc_internal(bpc, region, days_back, valuation_method, api_client)
    
    def _value_single_bpc_threadsafe(self, bpc: BPC, region: str, days_back: int, valuation_method: str, bpc_key: str) -> tuple:
        """
        Thread-safe wrapper for valuing a single BPC.
        
        Returns:
            Tuple of (bpc_key, valuation_result_or_exception)
        """
        try:
            # Rate limit API requests
            self.rate_limiter.wait_if_needed()
            api_client = self._get_thread_client()
            
            # Check cache first
            with self._cache_lock:
                if bpc_key in self._valuation_cache:
                    return bpc_key, self._valuation_cache[bpc_key]
            
            # Perform valuation
            result = self._value_single_bpc_internal(bpc, region, days_back, valuation_method, api_client)
            
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
        valuation_method: str,
        api_client: Adam4EveHistoricalClient
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
        name_key = bpc.name.strip().lower()
        item_id = bpc.item_id
        if not item_id:
            with self._html_cache_lock:
                item_id = self._item_id_cache.get(name_key)
        if not item_id:
            item_id = api_client.search_bpc_by_name(bpc.name)
            if item_id:
                with self._html_cache_lock:
                    self._item_id_cache[name_key] = item_id
        if not item_id:
            raise ValueError(f"Could not find item ID for BPC: {bpc.name}")
        logger.debug("Resolved item_id=%s for %s (cache hit: %s)", item_id, bpc.name, item_id == bpc.item_id)
        
        # Defer contract page fetch until needed (avoids Adam4EVE calls when using snapshot)
        cache_key = (item_id, region, days_back)
        with self._html_cache_lock:
            contract_html = self._contract_html_cache.get(cache_key)
        if not contract_html and not self.snapshot_client:
            with self._html_cache_lock:
                contract_html = self._contract_html_cache.get(cache_key)
            if not contract_html:
                try:
                    logger.debug("Fetching contract page for key=%s (thread=%s)", cache_key, threading.current_thread().name)
                    contract_html = self.api_client._get_contract_page_html(item_id, region, days_back)
                    with self._html_cache_lock:
                        self._contract_html_cache[cache_key] = contract_html
                except Exception:
                    contract_html = None
        else:
            if contract_html:
                logger.debug("Using cached contract page for key=%s", cache_key)

        # Determine if we should use efficiency matching or default pricing
        use_efficiency_matching = not (bpc.me_level == 0 and bpc.te_level == 0 and bpc.runs == 1)
        target_efficiency = None

        if use_efficiency_matching:
            # Use efficiency matching to find the closest column
            from src.models.bpc import BPCEfficiency
            target_efficiency = BPCEfficiency(bpc.me_level, bpc.te_level, bpc.runs)
            historical_prices = []
            matched_efficiency = None

            # Try snapshot first
            region_id = self._region_name_to_id(region)
            if self.snapshot_client:
                try:
                    snapshot_prices = self.snapshot_client.get_prices(
                        type_id=item_id,
                        region_id=region_id,
                        me_level=bpc.me_level,
                        te_level=bpc.te_level,
                        runs=bpc.runs
                    )
                    if snapshot_prices:
                        historical_prices = snapshot_prices
                        matched_efficiency = str(target_efficiency)
                        matched_runs = bpc.runs
                        contract_html = None  # avoid Adam4EVE usage
                        logger.debug("Using Everef snapshot for %s", bpc.name)
                except Exception as exc:
                    logger.debug("Everef snapshot lookup failed for %s: %s", bpc.name, exc)

            if not historical_prices:
                historical_prices, matched_efficiency = api_client.get_historical_prices_with_efficiency(
                    item_id, target_efficiency, region, days_back, html_content=contract_html
                )
                if matched_efficiency:
                    print(f"Using efficiency matching: {target_efficiency} -> {matched_efficiency}")
                else:
                    print(f"No efficiency match found, falling back to default pricing")
                    historical_prices = api_client.get_historical_prices(item_id, region, days_back, html_content=contract_html)
                    matched_efficiency = None
        else:
            # Use default pricing (first available price column)
            region_id = self._region_name_to_id(region)
            if self.snapshot_client:
                try:
                    historical_prices = self.snapshot_client.get_prices(
                        type_id=item_id,
                        region_id=region_id
                    )
                    if historical_prices:
                        matched_efficiency = None
                        contract_html = None
                        logger.debug("Using Everef snapshot (no efficiency match) for %s", bpc.name)
                    else:
                        historical_prices = api_client.get_historical_prices(item_id, region, days_back, html_content=contract_html)
                except Exception as exc:
                    logger.debug("Everef snapshot lookup failed for %s: %s", bpc.name, exc)
                    historical_prices = api_client.get_historical_prices(item_id, region, days_back, html_content=contract_html)
            else:
                historical_prices = api_client.get_historical_prices(item_id, region, days_back, html_content=contract_html)
            matched_efficiency = None
        
        # Calculate price statistics
        price_stats = self.calculate_price_statistics(historical_prices)
        
        # Determine fair market value based on method
        apply_multiplier = matched_efficiency is None  # avoid double-counting when we already matched efficiency
        efficiency_adjustment = 1.0
        matched_runs = None
        if matched_efficiency:
            from src.models.bpc import BPCEfficiency
            matched_eff_obj = BPCEfficiency.parse(matched_efficiency)
            if target_efficiency:
                efficiency_adjustment = self._calculate_efficiency_adjustment(target_efficiency, matched_eff_obj)
            matched_runs = matched_eff_obj.runs
            logger.debug(
                "Efficiency match found: target=%s matched=%s adj=%.4f",
                target_efficiency, matched_eff_obj, efficiency_adjustment
            )
        elif use_efficiency_matching:
            # No matching column found; assume best-available (perfect) column and adjust downward if needed
            from src.models.bpc import BPCEfficiency
            assumed_best = BPCEfficiency(10, 20, bpc.runs)
            efficiency_adjustment = self._calculate_efficiency_adjustment(target_efficiency, assumed_best)
            apply_multiplier = False  # avoid boosting off a perfect column baseline
            matched_runs = assumed_best.runs
            logger.debug(
                "No efficiency column matched; assuming best column %s and applying adj=%.4f",
                assumed_best, efficiency_adjustment
            )
        else:
            logger.debug("No efficiency match data; applying baseline multiplier for %s", bpc.name)

        blended_base = None
        try:
            # Reuse a single contract page fetch for all columns
            contract_html = api_client._get_contract_page_html(item_id, region, days_back)
            blended_base = self._blended_efficiency_baseline(
                item_id=item_id,
                region=region,
                days_back=days_back,
                target_efficiency=target_efficiency if use_efficiency_matching else None,
                api_client=api_client,
                valuation_method=valuation_method,
                contract_html=contract_html
            )
        except Exception as exc:
            logger.debug("Blended baseline failed for %s: %s", bpc.name, exc)

        fair_value = self._calculate_fair_value(
            historical_prices,
            bpc,
            valuation_method,
            apply_efficiency_multiplier=apply_multiplier,
            efficiency_adjustment=efficiency_adjustment,
            blended_base=blended_base,
            matched_runs=matched_runs
        )
        
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
        method: str,
        apply_efficiency_multiplier: bool = True,
        efficiency_adjustment: float = 1.0,
        blended_base: Optional[float] = None,
        matched_runs: Optional[int] = None
    ) -> float:
        """
        Calculate fair market value using specified method.
        
        Args:
            historical_prices: Historical price data
            bpc: BPC object being valued
            method: Valuation method to use
            apply_efficiency_multiplier: Whether to apply ME/TE multiplier (skip if column already matched)
            efficiency_adjustment: Small premium/discount when matched efficiency is close but not exact
            blended_base: Optional per-run blended baseline across efficiency columns
            matched_runs: Runs count from matched efficiency column (if any)
            
        Returns:
            Fair market value in ISK
        """
        if not historical_prices:
            return 0.0
        
        # For ask-driven data, trim deeper and prefer robust stats
        trim_fraction = 0.2
        trimmed_prices, trimmed_values = self._trim_price_series(historical_prices, trim_fraction=trim_fraction)
        prices = trimmed_values
        
        if not prices:
            return 0.0
        
        if blended_base is not None:
            base_value = blended_base  # per-run baseline already
        elif method == "median":
            base_value = statistics.median(prices)
        elif method == "mean":
            base_value = statistics.mean(prices)
        elif method == "weighted_median":
            # Weight recent prices more heavily
            base_value = self._weighted_median(trimmed_prices)
        elif method == "exponential_weighted":
            # Exponentially-weighted average with longer half-life to reduce single-day undercuts
            base_value = self._exponential_weighted_average(trimmed_prices, half_life_days=20)
        elif method == "conservative":
            # Use 25th percentile for conservative estimate
            sorted_prices = sorted(prices)
            percentile_25 = sorted_prices[len(sorted_prices) // 4]
            base_value = percentile_25
        else:
            # Default to weighted median for robustness on ask data
            base_value = self._weighted_median(trimmed_prices)
        
        # Normalize to per-run baseline
        runs_for_baseline = matched_runs or bpc.runs or 1
        if blended_base is not None:
            per_run_base = base_value  # already per-run
            base_total_for_log = per_run_base * runs_for_baseline
        else:
            base_total_for_log = base_value
            per_run_base = base_value / runs_for_baseline
        
        # Apply efficiency adjustments
        efficiency_multiplier = self._get_efficiency_multiplier(bpc) if apply_efficiency_multiplier else 1.0
        run_adjustment = self._run_count_adjustment(bpc.runs)
        
        adjusted_value = per_run_base * bpc.runs * efficiency_multiplier * efficiency_adjustment * run_adjustment
        logger.debug(
            "Fair value calc: base_total=%.2f per_run=%.2f multiplier=%.4f eff_adj=%.4f run_adj=%.4f result=%.2f (method=%s)",
            base_total_for_log,
            per_run_base,
            efficiency_multiplier,
            efficiency_adjustment,
            run_adjustment,
            adjusted_value,
            method
        )
        return adjusted_value
    
    def _region_name_to_id(self, region: str) -> int:
        region_mapping = {
            "The Forge": 10000002,
            "Domain": 10000043,
            "Sinq Laison": 10000032,
            "Heimatar": 10000030,
            "Metropolis": 10000042
        }
        return region_mapping.get(region, 10000002)
    
    def _preload_contract_pages(self, bpcs: List[BPC], region: str, days_back: int):
        """
        Resolve item IDs and fetch contract pages once per unique BPC name.
        """
        unique_names = {}
        for bpc in bpcs:
            key = bpc.name.strip().lower()
            if key not in unique_names:
                unique_names[key] = bpc.name
        
        def _load(name: str):
            item_id = self.api_client.search_bpc_by_name(name)
            if item_id:
                name_key = name.strip().lower()
                with self._html_cache_lock:
                    self._item_id_cache[name_key] = item_id
                try:
                    html = self.api_client._get_contract_page_html(item_id, region, days_back)
                    cache_key = (item_id, region, days_back)
                    with self._html_cache_lock:
                        self._contract_html_cache[cache_key] = html
                except Exception as exc:
                    logger.debug("Cache preload failed for %s: %s", name, exc)
            return name, item_id
        
        max_workers = min(self.max_workers, max(1, len(unique_names)))
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load, n): n for n in unique_names.values()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, item_id = future.result()
                    results[name.strip().lower()] = item_id
                except Exception as exc:
                    logger.debug("Preload error for %s: %s", name, exc)
        
        # Propagate resolved IDs back into BPC objects
        for bpc in bpcs:
            key = bpc.name.strip().lower()
            if key in results and results[key]:
                bpc.item_id = results[key]
    
    def _trim_price_series(
        self,
        historical_prices: List[HistoricalPrice],
        trim_fraction: float = 0.1
    ) -> Tuple[List[HistoricalPrice], List[float]]:
        """
        Drop the lowest tail of prices to reduce outlier undercuts.
        Returns trimmed HistoricalPrice list and corresponding price floats.
        """
        valid_prices = [p for p in historical_prices if p.price > 0]
        if not valid_prices:
            return [], []
        
        # Sort by price ascending for trimming
        sorted_by_price = sorted(valid_prices, key=lambda x: x.price)
        trim_count = int(len(sorted_by_price) * trim_fraction)
        
        if trim_count >= len(sorted_by_price):
            trimmed = sorted_by_price
        else:
            trimmed = sorted_by_price[trim_count:]
        
        if not trimmed:
            trimmed = sorted_by_price
        
        if trim_count > 0:
            logger.debug(
                "Trimmed %d/%d price points (%.1f%%) from low tail; first kept price=%.2f",
                trim_count,
                len(sorted_by_price),
                trim_fraction * 100,
                trimmed[0].price
            )
        
        return trimmed, [p.price for p in trimmed]

    def _blended_efficiency_baseline(
        self,
        item_id: int,
        region: str,
        days_back: int,
        target_efficiency,
        api_client: Adam4EveHistoricalClient,
        valuation_method: str,
        contract_html: Optional[str] = None
    ) -> Optional[float]:
        """
        Build a blended baseline across efficiency columns, weighting by volume and closeness.
        """
        from src.models.bpc import BPCEfficiency
        efficiency_columns = api_client.get_efficiency_columns(item_id, contract_html)
        if not efficiency_columns:
            return None
        
        target_eq = target_efficiency.calculate_equivalent_runs() if target_efficiency else None
        
        bases = []
        weights = []
        
        for eff_str, col_idx in efficiency_columns.items():
            try:
                eff_obj = BPCEfficiency.parse(eff_str)
            except ValueError:
                continue
            
            try:
                price_series = api_client._parse_contract_price_html_with_column(
                    item_id, region, col_idx, days_back=days_back, html_content=contract_html
                )
            except Exception as exc:
                logger.debug("Failed to parse column %s for item %s: %s", eff_str, item_id, exc)
                continue
            
            trimmed, trimmed_values = self._trim_price_series(price_series, trim_fraction=0.2)
            if not trimmed_values:
                continue
            
            # Column base using chosen method
            if valuation_method == "median":
                col_base = statistics.median(trimmed_values)
            elif valuation_method == "mean":
                col_base = statistics.mean(trimmed_values)
            elif valuation_method == "weighted_median":
                col_base = self._weighted_median(trimmed)
            elif valuation_method == "exponential_weighted":
                col_base = self._exponential_weighted_average(trimmed, half_life_days=20)
            elif valuation_method == "conservative":
                sorted_prices = sorted(trimmed_values)
                col_base = sorted_prices[len(sorted_prices) // 4]
            else:
                col_base = self._weighted_median(trimmed)
            
            if eff_obj.runs <= 0:
                continue
            col_base_per_run = col_base / eff_obj.runs
            
            # Volume proxy (sum of volumes; fallback to count)
            vol = sum(p.volume for p in trimmed if p.volume > 0)
            if vol <= 0:
                vol = len(trimmed)
            
            # Closeness weight
            if target_eq is not None:
                delta = abs(target_eq - eff_obj.calculate_equivalent_runs())
                closeness = 1 / (1 + delta)
            else:
                closeness = 1.0
            
            weight = (vol ** 0.5) * closeness
            if weight > 0 and col_base_per_run > 0:
                bases.append(col_base_per_run)
                weights.append(weight)
                logger.debug(
                    "Blended baseline component: eff=%s base_total=%.2f per_run=%.2f vol=%s weight=%.4f closeness=%.4f",
                    eff_str, col_base, col_base_per_run, vol, weight, closeness
                )
        
        if not bases or sum(weights) == 0:
            return None
        
        blended = sum(b * w for b, w in zip(bases, weights)) / sum(weights)
        logger.debug("Blended baseline per-run value: %.2f using %d columns", blended, len(bases))
        return blended
    
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
    
    def _calculate_efficiency_adjustment(self, target_efficiency, matched_efficiency) -> float:
        """
        Small premium/discount when the matched efficiency column differs slightly from the target.
        Uses equivalent runs delta with gentle scaling and caps at +/-10%.
        """
        try:
            matched_eq = matched_efficiency.calculate_equivalent_runs()
            target_eq = target_efficiency.calculate_equivalent_runs()
            
            if matched_eq <= 0:
                return 1.0
            
            delta = (target_eq - matched_eq) / matched_eq  # positive if target is better than matched
            
            k = 0.25
            adjustment = delta * k
            min_mag = 0.03
            if delta != 0 and abs(adjustment) < min_mag:
                adjustment = min_mag * (1 if delta > 0 else -1)
            
            adjustment = max(-0.1, min(0.1, adjustment))  # cap at +/-10%
            adjusted = 1.0 + adjustment
            logger.debug(
                "Efficiency adjustment: target_eq=%.4f matched_eq=%.4f delta=%.4f adj=%.4f final=%.4f",
                target_eq, matched_eq, delta, adjustment, adjusted
            )
            return adjusted
        except Exception:
            return 1.0

    def _run_count_adjustment(self, runs: int) -> float:
        """
        Small, bounded adjustment for run count to reflect slot throughput/friction.
        Premium for lower runs (parallelism/flexibility), slight discount for very high runs (bulk friction).
        """
        if runs <= 0:
            return 1.0
        delta = (5 - runs) / 50.0  # runs=1 -> +0.08, runs=10 -> -0.1 (then clamped)
        delta = max(-0.05, min(0.05, delta))
        return 1.0 + delta

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
        
        effective_workers = self.max_workers
        print(f"Processing {len(bpcs)} BPCs using {effective_workers} threads (max {self.rate_limiter.max_requests_per_second} req/s)")
        
        # Preload item IDs and contract pages per unique name to avoid duplicate calls
        if not self.snapshot_client:
            self._preload_contract_pages(bpcs, region, days_back)
        
        # Create tasks for thread pool
        tasks = []
        for bpc in bpcs:
            bpc_key = f"{bpc.name}|{bpc.me_level}|{bpc.te_level}|{bpc.runs}"
            tasks.append((bpc, region, days_back, valuation_method, bpc_key))
        
        results = []
        failed_count = 0
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
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
