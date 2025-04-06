#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:42
#  Updated: 06/04/2025 13:42


import json
from datetime import datetime, timedelta
from typing import Dict, Any


import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class InputData:
    tickers: List[str]
    end_date: str
    start_date: str
    in_sample_days: int
    computed_start_date: str
    metadata: Dict[str, Any]

def read_input(file_path: str) -> InputData:
        """
        Reads and processes the tickers configuration JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            ConfigData: An object containing all configuration data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If invalid JSON
            KeyError: If required fields are missing
        """
        with open(file_path, 'r') as f:
            config = json.load(f)

        # Validate required fields
        if 'tickers' not in config or 'dates' not in config:
            raise KeyError("Configuration must contain 'tickers' and 'dates' fields")

        # Compute dates
        end_date = datetime.strptime(config['dates']['end_date'], '%Y-%m-%d')
        start_date = end_date - timedelta(days=config['dates']['in-sample-days'])
        computed_start_date = start_date.strftime('%Y-%m-%d')

        return InputData(
            tickers=config['tickers'],
            end_date=config['dates']['end_date'],
            start_date=start_date.strftime('%Y-%m-%d'),
            in_sample_days=config['dates']['in-sample-days'],
            computed_start_date=computed_start_date,
            metadata=config.get('metadata', {})
        )

def inspect_input(config: InputData) -> None:
    """
    Prints a human-readable inspection of the configuration.

    Args:
        config: ConfigData object returned by read_input()
    """
    if not isinstance(config, InputData):
        raise ValueError("Input must be a ConfigData object")

    # Header
    print("\n" + "=" * 50)
    print("INPUT PARAMETERS")
    print("=" * 50)

    # Tickers section
    print(f"\n[STOCK TICKERS]\nCount: {len(config.tickers)}")
    print(f"Tickers: {', '.join(config.tickers[:10])}{'...' if len(config.tickers) > 10 else ''}")

    # Dates section
    print(f"\n[DATE RANGE]")
    print(f"End Date:       {config.end_date}")
    print(f"Start Date:     {config.start_date}")
    print(f"Computed Start: {config.computed_start_date}")
    print(f"In-Sample Days: {config.in_sample_days}")

    # Metadata section
    if config.metadata:
        print("\n[METADATA]")
        for k, v in config.metadata.items():
            print(f"{k.replace('_', ' ').title()}: {v}")

    print("\n" + "=" * 50 + "\n")