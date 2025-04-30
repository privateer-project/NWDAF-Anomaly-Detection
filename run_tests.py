#!/usr/bin/env python3
"""
Run tests for the alert filter model.

This script runs the unit tests for the alert filter model to ensure that
everything is working correctly.

Usage:
    python run_tests.py
"""

import unittest
import sys
from tests.test_alert_filter import TestAlertFilter

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAlertFilter)
    
    # Run the tests
    result = unittest.TextTestRunner().run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
