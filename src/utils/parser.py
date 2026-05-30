import argparse

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-scale clouds reconstruction")
    parser.add_argument("nside", type=int, help="HEALPix NSIDE resolution parameter")
    return parser.parse_args()