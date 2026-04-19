"""
Test script for Fitbit integration
Run this script to verify the Fitbit integration components are working properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.fitbit_auth import FitbitOAuth
from app.fitbit_api import FitbitAPI
from app.data_merger import DataMerger
from datetime import date

def test_fitbit_oauth():
    """Test Fitbit OAuth functionality"""
    print("Testing Fitbit OAuth...")
    
    # Test authorization URL generation
    try:
        auth_url = FitbitOAuth.get_authorization_url()
        print(f"  Authorization URL generated: {auth_url[:50]}...")
        assert "fitbit.com" in auth_url
        print("  OAuth URL generation: PASSED")
    except Exception as e:
        print(f"  OAuth URL generation: FAILED - {e}")
        return False
    
    return True

def test_data_merger():
    """Test data merging functionality"""
    print("Testing Data Merger...")
    
    try:
        # Test manual data only
        user_data = {
            'sleep_duration': 7.5,
            'sleep_latency': 20,
            'wake_count': 2
        }
        
        merged = DataMerger.merge_sleep_data(user_data, None, date.today())
        assert merged['sleep_duration'] == 7.5
        assert merged['_metadata']['data_sources']['sleep_duration'] == 'manual'
        print("  Manual data merging: PASSED")
        
        # Test Fitbit data only
        fitbit_data = {
            'total_minutes_asleep': 480,
            'sleep_efficiency': 85,
            'sleep_stages': {
                'deep': 90,
                'rem': 105,
                'wake': 30
            }
        }
        
        merged = DataMerger.merge_sleep_data({}, fitbit_data, date.today())
        assert merged['sleep_duration'] == 8.0  # 480/60
        assert merged['_metadata']['data_sources']['sleep_duration'] == 'fitbit'
        print("  Fitbit data merging: PASSED")
        
        # Test mixed data (manual overrides Fitbit)
        merged = DataMerger.merge_sleep_data(
            {'sleep_duration': 6.5},  # Manual entry
            fitbit_data,             # Fitbit data (8.0 hours)
            date.today()
        )
        assert merged['sleep_duration'] == 6.5  # Manual takes priority
        assert merged['_metadata']['data_sources']['sleep_duration'] == 'manual'
        print("  Mixed data merging: PASSED")
        
    except Exception as e:
        print(f"  Data merging: FAILED - {e}")
        return False
    
    return True

def test_fitbit_api_formatting():
    """Test Fitbit API data formatting"""
    print("Testing Fitbit API data formatting...")
    
    try:
        # Test _format_fitbit_for_form method
        fitbit_data = {
            'total_minutes_asleep': 450,
            'sleep_efficiency': 88,
            'sleep_stages': {
                'deep': 90,
                'rem': 112,
                'wake': 24
            },
            'sleep_start_time': '2023-12-01T23:30:00.000Z',
            'sleep_end_time': '2023-12-02T07:00:00.000Z'
        }
        
        formatted = DataMerger._format_fitbit_for_form(fitbit_data)
        
        assert formatted['sleep_duration'] == 7.5
        assert formatted['sleep_efficiency'] == 88
        assert formatted['deep_sleep_percent'] == 20  # 90/450 * 100
        assert formatted['rem_sleep_percent'] == 25   # 112/450 * 100
        assert formatted['has_smartwatch'] == True
        print("  Fitbit data formatting: PASSED")
        
    except Exception as e:
        print(f"  Fitbit data formatting: FAILED - {e}")
        return False
    
    return True

def test_field_source_tracking():
    """Test field source tracking functionality"""
    print("Testing field source tracking...")
    
    try:
        user_data = {'sleep_duration': 7.0}
        fitbit_data = {
            'total_minutes_asleep': 480,
            'sleep_stages': {'deep': 96, 'rem': 96}
        }
        
        sources = DataMerger._get_field_sources(user_data, fitbit_data)
        
        assert sources['sleep_duration'] == 'manual'  # User overrides
        assert sources['deep_sleep_percent'] == 'fitbit'  # Only in Fitbit
        assert sources['sleep_latency'] == 'default'  # In neither
        print("  Field source tracking: PASSED")
        
    except Exception as e:
        print(f"  Field source tracking: FAILED - {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("FITBIT INTEGRATION TESTS")
    print("=" * 50)
    
    tests = [
        test_fitbit_oauth,
        test_data_merger,
        test_fitbit_api_formatting,
        test_field_source_tracking
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests PASSED! Fitbit integration is ready.")
        return True
    else:
        print("Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
