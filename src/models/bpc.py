"""
Blueprint Copy (BPC) Data Models

This module contains data models for EVE Online Blueprint Copies and their valuation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class BPCType(Enum):
    """Types of Blueprint Copies."""
    SHIP = "Ship"
    MODULE = "Module" 
    AMMUNITION = "Ammunition"
    STRUCTURE = "Structure"
    RIG = "Rig"
    OTHER = "Other"


@dataclass
class BPCEfficiency:
    """Represents BPC efficiency ratings in ME/TE/Runs format."""
    material_efficiency: int  # ME level (0-10)
    time_efficiency: int      # TE level (0-20)  
    runs: int                 # Number of runs
    
    def __post_init__(self):
        """Validate efficiency values."""
        if not 0 <= self.material_efficiency <= 10:
            raise ValueError(f"Material Efficiency must be 0-10, got {self.material_efficiency}")
        if not 0 <= self.time_efficiency <= 20:
            raise ValueError(f"Time Efficiency must be 0-20, got {self.time_efficiency}")
        if self.runs <= 0:
            raise ValueError(f"Runs must be positive, got {self.runs}")
    
    @classmethod
    def parse(cls, efficiency_str: str) -> 'BPCEfficiency':
        """
        Parse efficiency string in format 'A/B/C' into BPCEfficiency.
        
        Args:
            efficiency_str: String in format "ME/TE/Runs" (e.g., "10/20/1")
            
        Returns:
            BPCEfficiency object
            
        Raises:
            ValueError: If string format is invalid
        """
        try:
            parts = efficiency_str.split('/')
            if len(parts) != 3:
                raise ValueError(f"Expected format 'ME/TE/Runs', got '{efficiency_str}'")
            
            me = int(parts[0])
            te = int(parts[1]) 
            runs = int(parts[2])
            
            return cls(me, te, runs)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid efficiency format '{efficiency_str}': {e}")
    
    def calculate_equivalent_runs(self) -> float:
        """
        Calculate equivalent runs using the formula: N = C/((1-A/100)(1-B/100))
        where A=ME, B=TE, C=runs.
        
        This represents the effective output considering efficiency bonuses.
        
        Returns:
            Equivalent runs as a float
        """
        me_factor = 1 - (self.material_efficiency / 100)
        te_factor = 1 - (self.time_efficiency / 100)
        
        # Avoid division by zero
        if me_factor <= 0 or te_factor <= 0:
            return float('inf')
            
        return self.runs / (me_factor * te_factor)
    
    def __str__(self) -> str:
        """String representation in ME/TE/Runs format."""
        return f"{self.material_efficiency}/{self.time_efficiency}/{self.runs}"


@dataclass
class BPC:
    """Represents a Blueprint Copy with its attributes."""
    name: str
    item_id: int
    me_level: int  # Material Efficiency (0-10)
    te_level: int  # Time Efficiency (0-20)
    runs: int      # Number of runs remaining
    bpc_type: BPCType = BPCType.OTHER
    notes: str = ""
    quantity: int = 1  # Number of this BPC we have
    
    @property
    def efficiency_rating(self) -> str:
        """Get a human-readable efficiency rating."""
        if self.me_level == 10 and self.te_level == 20:
            return "Perfect"
        elif self.me_level >= 9 and self.te_level >= 18:
            return "Excellent"
        elif self.me_level >= 7 and self.te_level >= 14:
            return "Good"
        elif self.me_level >= 5 and self.te_level >= 10:
            return "Average"
        else:
            return "Poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'item_id': self.item_id,
            'me_level': self.me_level,
            'te_level': self.te_level,
            'runs': self.runs,
            'bpc_type': self.bpc_type.value,
            'efficiency_rating': self.efficiency_rating,
            'notes': self.notes
        }


@dataclass
class HistoricalPrice:
    """Represents historical price data for a BPC."""
    date: datetime
    price: float
    volume: int
    region: str
    
    
@dataclass
class PriceStatistics:
    """Statistical analysis of BPC prices."""
    mean_price: float
    median_price: float
    min_price: float
    max_price: float
    std_deviation: float
    sample_size: int
    price_trend: str  # "rising", "falling", "stable"
    confidence_score: float  # 0-100, based on sample size and consistency


@dataclass
class BPCValuation:
    """Complete valuation analysis for a BPC."""
    bpc: BPC
    fair_market_value: float
    valuation_method: str  # "historical_median", "weighted_average", etc.
    price_statistics: PriceStatistics
    historical_data: List[HistoricalPrice]
    region: str
    analysis_date: datetime
    confidence_level: str  # "High", "Medium", "Low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'bpc': self.bpc.to_dict(),
            'fair_market_value': self.fair_market_value,
            'valuation_method': self.valuation_method,
            'price_statistics': {
                'mean_price': self.price_statistics.mean_price,
                'median_price': self.price_statistics.median_price,
                'min_price': self.price_statistics.min_price,
                'max_price': self.price_statistics.max_price,
                'std_deviation': self.price_statistics.std_deviation,
                'sample_size': self.price_statistics.sample_size,
                'price_trend': self.price_statistics.price_trend,
                'confidence_score': self.price_statistics.confidence_score
            },
            'region': self.region,
            'analysis_date': self.analysis_date.isoformat(),
            'confidence_level': self.confidence_level
        }