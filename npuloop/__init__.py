"""npuloop — closed-loop NPU-aware model compression toolkit.

virtual NPU cost model  ->  static readiness lint  ->  quantization / pruning / surgery
        ^                                                         |
        +--------------- bit-exact integer engine (NumPy + C++) ---+
"""
__version__ = "0.1.0"
