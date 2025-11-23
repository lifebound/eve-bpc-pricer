"""
CSV Parser for BPC Data

This module handles parsing CSV files containing BPC information
and converting them to BPC objects for valuation.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.models.bpc import BPC, BPCType
from src.models.bpc import BPCEfficiency


class BPCCSVParser:
    """Parser for BPC data from CSV files."""
    
    def __init__(self):
        """Initialize the CSV parser."""
        self.required_columns = ['bpc_name']  # Only BPC name is truly required
        self.optional_columns = ['bpc_type', 'notes', 'item_id', 'bpc_efficiency', 'me_level', 'te_level', 'runs']
    
    def _standardize_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame columns to handle different input formats.
        
        Handles:
        - Single column: "BPC Name" or "BPC Name Quantity" 
        - Multi-column tab-separated: Name, Quantity, Type, Category, Description
        - Standard CSV with headers
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            DataFrame with standardized column names
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Case 1: Single column format
        if len(df.columns) == 1:
            first_col_name = df.columns[0]
            first_value = str(df.iloc[0, 0]).strip() if len(df) > 0 else ""
            
            is_likely_header = self._looks_like_header(first_col_name)
            first_val_is_header = self._looks_like_header(first_value)

            if is_likely_header:
                # Column header looks like "name" / "bpc_name" / etc. Just rename it.
                df = df.rename(columns={first_col_name: 'bpc_name'})
            elif first_val_is_header:
                # First *row* looks like header: drop that row and rename column
                df = df.iloc[1:].copy()
                df = df.rename(columns={first_col_name: 'bpc_name'})
            else:
                # Treat as header-less single-column: rename column
                df = df.rename(columns={first_col_name: 'bpc_name'})
                
        # Case 2: Multi-column tab-separated format (specific pattern)
        elif len(df.columns) >= 5 and 'bpc_name' not in df.columns and not any(self._looks_like_header(col) for col in df.columns):
            # Tab-separated format: Name, Quantity, Type, Category, Description
            column_mapping = {
                df.columns[0]: 'bpc_name',
                df.columns[1]: 'explicit_quantity', 
                df.columns[2]: 'bpc_type_hint',
                df.columns[3]: 'category',
                df.columns[4]: 'description'
            }
            df = df.rename(columns=column_mapping)
            
        # Case 3: Standard CSV format with headers
        elif 'bpc_name' not in df.columns and len(df.columns) > 1:
            # Try to intelligently map columns
            for col in df.columns:
                if self._looks_like_header(col):
                    df = df.rename(columns={col: 'bpc_name'})
                    break
                    
        return df
    
    def _looks_like_header(self, text: Any) -> bool:
        if not isinstance(text, str):
            return False
        
        t = text.strip().lower()
        if not t:
            return False
    
        # Canonical header candidates
        header_tokens = {
            'bpc_name',
            'name',
            'blueprint',
            'bpc',
            'item',
            'item_name',
            'blueprint_name',
        }
    
        # Normalize a bit
        t_nospace = t.replace(' ', '').replace('_', '')
        
        # Exact-ish matching, not substring
        if t in header_tokens:
            return True
        if t_nospace in header_tokens:
            return True
        if t.endswith(' name') or t.endswith('_name'):
            return True
        
        return False

    def _parse_bpc_name_and_quantity(self, name_value: str) -> tuple[str, int]:
        """
        Parse BPC name and quantity from a string.
        Handles formats like:
        - "Raven Blueprint 4" -> ("Raven Blueprint", 4)
        - "Raven Blueprint" -> ("Raven Blueprint", 1)
        
        Args:
            name_value: BPC name string potentially with trailing quantity
            
        Returns:
            Tuple of (bpc_name, quantity)
        """
        if not name_value:
            return "", 1
            
        name_value = str(name_value).strip()
        
        # Split on whitespace and check if last token is a number
        parts = name_value.rsplit(' ', 1)
        
        if len(parts) == 2:
            name_part, potential_quantity = parts
            try:
                quantity = int(potential_quantity)
                if quantity > 0:
                    return name_part.strip(), quantity
            except ValueError:
                pass
        
        # If we can't parse quantity, return name as-is with quantity 1
        return name_value, 1

    def _parse_efficiency_from_description(self, description: str) -> tuple[int, int, int]:
        """
        Parse ME/TE/Runs from description text like:
        'BLUEPRINT COPY - Runs: 2 - Material Efficiency: 0 - Time Efficiency: 0'
        
        Args:
            description: Description string potentially containing efficiency data
            
        Returns:
            Tuple of (me_level, te_level, runs)
        """
        if not description or pd.isna(description):
            return 0, 0, 1
            
        description = str(description).strip()
        
        # Default values
        me_level, te_level, runs = 0, 0, 1
        
        # Try to extract ME level
        import re
        me_match = re.search(r'Material Efficiency:\s*(\d+)', description, re.IGNORECASE)
        if me_match:
            me_level = int(me_match.group(1))
            
        # Try to extract TE level  
        te_match = re.search(r'Time Efficiency:\s*(\d+)', description, re.IGNORECASE)
        if te_match:
            te_level = int(te_match.group(1))
            
        # Try to extract Runs
        runs_match = re.search(r'Runs:\s*(\d+)', description, re.IGNORECASE)
        if runs_match:
            runs = int(runs_match.group(1))
            
        return me_level, te_level, runs

    def parse_csv_file(self, file_path: str) -> List[BPC]:
        """
        Parse BPC data from a CSV file.
        Supports multiple formats:
        - Simple single column: "BPC Name" or "BPC Name Quantity"
        - Tab-separated with description: "Name\tQty\tType\tCategory\tDescription"
        - Standard CSV with headers
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of BPC objects
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            # First, try to detect the delimiter by reading a sample
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                
            # Detect delimiter
            delimiter = ','
            if '\t' in sample and sample.count('\t') > sample.count(','):
                delimiter = '\t'
                
            # Read CSV file with detected delimiter
            df = pd.read_csv(file_path, delimiter=delimiter, dtype=str, keep_default_na=False)
            
            # Handle different column structures
            df = self._standardize_dataframe_columns(df)
            
            # Validate required columns
            self._validate_columns(df)
            
            # Convert to BPC objects
            bpcs = []
            for _, row in df.iterrows():
                bpc = self._row_to_bpc(row)
                if bpc:
                    bpcs.append(bpc)
            
            return bpcs
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error reading CSV: {e}")
    
    def _validate_columns(self, df: pd.DataFrame):
        """
        Validate that required columns are present.
        
        Args:
            df: Pandas DataFrame with CSV data
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    def _row_to_bpc(self, row: pd.Series) -> Optional[BPC]:
        """
        Convert a pandas Series row to a BPC object.
        Handles multiple formats:
        - Simple: Just BPC name (with optional trailing quantity)
        - Complex: BPC name + explicit quantity + description with embedded ME/TE/Runs
        
        Args:
            row: Pandas Series with BPC data
            
        Returns:
            BPC object or None if conversion fails
        """
        try:
            # Required field - BPC name with potential quantity parsing
            raw_bpc_name = str(row['bpc_name']).strip()
            if not raw_bpc_name or raw_bpc_name.startswith('#'):  # Skip comment rows
                return None
            
            # Check if we have explicit quantity column (various possible names)
            explicit_quantity = row.get('explicit_quantity') or row.get('quantity')
            if pd.notna(explicit_quantity) and str(explicit_quantity).strip():
                # Use explicit quantity and don't parse from name
                bpc_name = raw_bpc_name
                quantity = self._safe_parse_int(explicit_quantity, default=1, min_val=1)
            else:
                # Parse name and quantity from BPC name
                bpc_name, quantity = self._parse_bpc_name_and_quantity(raw_bpc_name)
            
            # Check if we have embedded efficiency data in description
            description = row.get('description', '')
            # Parse explicit efficiency string if present
            efficiency_str = row.get('bpc_efficiency', '')
            efficiency_hint = None
            if pd.notna(efficiency_str) and str(efficiency_str).strip():
                try:
                    efficiency_hint = BPCEfficiency.parse(str(efficiency_str).strip())
                except ValueError:
                    efficiency_hint = None

            if pd.notna(description) and 'Material Efficiency' in str(description):
                # Parse efficiency from description
                me_from_desc, te_from_desc, runs_from_desc = self._parse_efficiency_from_description(description)
                # Use description values as defaults, but allow explicit columns to override
                me_level = self._safe_parse_int(row.get('me_level'), default=me_from_desc, min_val=0, max_val=10)
                te_level = self._safe_parse_int(row.get('te_level'), default=te_from_desc, min_val=0, max_val=20)
                runs = self._safe_parse_int(row.get('runs'), default=runs_from_desc, min_val=1)
            elif efficiency_hint:
                # Use parsed efficiency string as defaults
                me_level = self._safe_parse_int(row.get('me_level'), default=efficiency_hint.material_efficiency, min_val=0, max_val=10)
                te_level = self._safe_parse_int(row.get('te_level'), default=efficiency_hint.time_efficiency, min_val=0, max_val=20)
                runs = self._safe_parse_int(row.get('runs'), default=efficiency_hint.runs, min_val=1)
            else:
                # Parse efficiency values with standard defaults
                me_level = self._safe_parse_int(row.get('me_level'), default=0, min_val=0, max_val=10)
                te_level = self._safe_parse_int(row.get('te_level'), default=0, min_val=0, max_val=20)
                runs = self._safe_parse_int(row.get('runs'), default=1, min_val=1)
            
            # Parse BPC type with hint from type column
            bpc_type_hint = row.get('bpc_type_hint', '')
            if pd.notna(bpc_type_hint) and 'Rig' in str(bpc_type_hint):
                bpc_type = BPCType.RIG
            else:
                bpc_type = self._parse_bpc_type(row.get('bpc_type', ''))
            
            # Other optional fields
            notes = str(row.get('notes', '')).strip()
            item_id = self._parse_item_id(row.get('item_id', 0))
            
            return BPC(
                name=bpc_name,
                item_id=item_id,
                me_level=me_level,
                te_level=te_level,
                runs=runs,
                bpc_type=bpc_type,
                notes=notes,
                quantity=quantity
            )
            
        except Exception as e:
            print(f"Error parsing row for '{row.get('bpc_name', 'unknown')}': {e}")
            return None
    
    def _safe_parse_int(self, value, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """
        Safely parse an integer value with defaults and validation.
        
        Args:
            value: Value to parse (can be string, int, float, or NaN)
            default: Default value if parsing fails
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Parsed and validated integer
        """
        try:
            # Handle NaN, empty strings, or None
            if pd.isna(value) or value == '' or value is None:
                return default
                
            # Try to convert to int
            parsed_value = int(float(str(value).strip()))
            
            # Apply min/max constraints
            if min_val is not None:
                parsed_value = max(min_val, parsed_value)
            if max_val is not None:
                parsed_value = min(max_val, parsed_value)
                
            return parsed_value
            
        except (ValueError, TypeError):
            return default
    
    def _parse_bpc_type(self, type_str: str) -> BPCType:
        """
        Parse BPC type from string.
        
        Args:
            type_str: String representation of BPC type
            
        Returns:
            BPCType enum value
        """
        if not type_str or pd.isna(type_str):
            return BPCType.OTHER
        
        type_mapping = {
            'ship': BPCType.SHIP,
            'module': BPCType.MODULE,
            'ammunition': BPCType.AMMUNITION,
            'ammo': BPCType.AMMUNITION,
            'structure': BPCType.STRUCTURE,
            'rig': BPCType.RIG,
            'other': BPCType.OTHER
        }
        
        normalized = str(type_str).lower().strip()
        return type_mapping.get(normalized, BPCType.OTHER)
    
    def _parse_item_id(self, item_id_value: Any) -> int:
        """
        Parse item ID from various input types.
        
        Args:
            item_id_value: Item ID value (int, float, str)
            
        Returns:
            Integer item ID, 0 if invalid
        """
        if pd.isna(item_id_value):
            return 0
        
        try:
            return int(float(item_id_value))
        except (ValueError, TypeError):
            return 0
    
    def export_bpcs_to_csv(self, bpcs: List[BPC], file_path: str):
        """
        Export BPC objects to a CSV file.
        
        Args:
            bpcs: List of BPC objects
            file_path: Output CSV file path
        """
        # Convert BPCs to dictionary format
        bpc_data = []
        for bpc in bpcs:
            bpc_dict = {
                'bpc_name': bpc.name,
                'item_id': bpc.item_id,
                'me_level': bpc.me_level,
                'te_level': bpc.te_level,
                'runs': bpc.runs,
                'quantity': bpc.quantity,
                'bpc_type': bpc.bpc_type.value,
                'efficiency_rating': bpc.efficiency_rating,
                'notes': bpc.notes
            }
            bpc_data.append(bpc_dict)
        
        # Create DataFrame and save
        df = pd.DataFrame(bpc_data)
        df.to_csv(file_path, index=False)
    
    def create_sample_csv(self, file_path: str = "sample_blueprints.csv"):
        """
        Create a sample CSV file with example BPC data.
        
        Args:
            file_path: Path for the sample CSV file
        """
        sample_data = [
            {
                'bpc_name': 'Raven Blueprint Copy',
                'item_id': 11567,
                'me_level': 10,
                'te_level': 20,
                'runs': 10,
                'bpc_type': 'Ship',
                'notes': 'Perfect BPC from research'
            },
            {
                'bpc_name': 'Dominix Blueprint Copy',
                'item_id': 11562,
                'me_level': 9,
                'te_level': 18,
                'runs': 5,
                'bpc_type': 'Ship',
                'notes': 'Nearly perfect'
            },
            {
                'bpc_name': 'Large Shield Extender II Blueprint Copy',
                'item_id': 0,  # Would need actual item ID
                'me_level': 8,
                'te_level': 16,
                'runs': 50,
                'bpc_type': 'Module',
                'notes': 'Good efficiency for modules'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False)
        print(f"Sample CSV created at: {file_path}")
    
    def create_blank_template(self, file_path: str = "blank_bpc_template.csv"):
        """
        Create a blank CSV template for users to fill out with their BPC data.
        
        Args:
            file_path: Path for the blank template CSV file
        """
        # Create template with headers and helpful comments
        template_data = {
            'bpc_name': ['# REQUIRED: Enter BPC name (e.g., "Raven Blueprint Copy")'],
            'item_id': ['# Optional: EVE item ID (leave blank if unknown)'],
            'me_level': ['# Optional: Material Efficiency 0-10 (default: 0)'],
            'te_level': ['# Optional: Time Efficiency 0-20 (default: 0)'],
            'runs': ['# Optional: Number of runs remaining (default: 1)'],
            'bpc_type': ['# Optional: Ship, Module, Ammunition, Structure, Rig, Other'],
            'notes': ['# Optional: Any notes about this BPC'],
            'bpc_efficiency': ['# Optional: Format as ME/TE/Runs (e.g., "10/20/1") for efficiency matching']
        }
        
        df = pd.DataFrame(template_data)
        df.to_csv(file_path, index=False)
        
        # Also create a completely empty version
        empty_file_path = file_path.replace('.csv', '_empty.csv')
        empty_data = {col: [] for col in template_data.keys()}
        empty_df = pd.DataFrame(empty_data)
        empty_df.to_csv(empty_file_path, index=False)
        
        print(f"Blank CSV template created at: {file_path}")
        print(f"Empty CSV template created at: {empty_file_path}")
        print(f"\nTemplate includes columns:")
        print(f"  Required: {', '.join(self.required_columns)}")
        print(f"  Optional: {', '.join(self.optional_columns)}, bpc_efficiency")
        print(f"\nFill out the template with your BPC data and use --csv to analyze.")


def validate_bpc_data(bpcs: List[BPC]) -> Dict[str, Any]:
    """
    Validate a list of BPC objects and return summary statistics.
    
    Args:
        bpcs: List of BPC objects to validate
        
    Returns:
        Dictionary with validation results and statistics
    """
    if not bpcs:
        return {
            'valid': False,
            'error': 'No BPC data provided',
            'count': 0
        }
    
    total_count = len(bpcs)
    total_quantity = sum(getattr(bpc, 'quantity', 1) for bpc in bpcs)
    type_counts = {}
    efficiency_stats = {'perfect': 0, 'excellent': 0, 'good': 0, 'average': 0, 'poor': 0}
    
    for bpc in bpcs:
        # Count by type
        type_key = bpc.bpc_type.value
        type_counts[type_key] = type_counts.get(type_key, 0) + 1
        
        # Count by efficiency
        rating = bpc.efficiency_rating.lower()
        if rating in efficiency_stats:
            efficiency_stats[rating] += 1
    
    return {
        'valid': True,
        'count': total_count,
        'total_quantity': total_quantity,
        'type_breakdown': type_counts,
        'efficiency_breakdown': efficiency_stats,
        'average_me': sum(bpc.me_level for bpc in bpcs) / total_count,
        'average_te': sum(bpc.te_level for bpc in bpcs) / total_count,
        'total_runs': sum(bpc.runs for bpc in bpcs)
    }
