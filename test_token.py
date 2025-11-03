"""
test_token.py: Quick script to test SAI API token validity.
This module provides functionality to validate SAI (CompetesAI) API tokens
by making a test request to the CompetesAI API endpoint.

Author: Saswata Sarkar
Email: sarkarsaswata01@gmail.com
Created: 2025

__license__ = "MIT"
__status__ = "Development"
"""

import os

import requests


def test_sai_token():
    token = os.environ.get('SAI_TOKEN')

    if not token:
        print("❌ SAI_TOKEN environment variable not set")
        return False

    print("✅ Token found and loaded from environment.")

    # Test API call
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'test-script'
    }

    url = 'https://api.competesai.com/v1/competitions/lower-t1-penalty-kick-goalie'

    print(f"\n🔍 Testing token against: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}")

        if response.status_code == 200:
            print("\n✅ Token is VALID! Authentication successful.")
            return True
        elif response.status_code == 401:
            print("\n❌ Token is INVALID or EXPIRED")
            print("   → Go to https://competesai.com and regenerate your API token")
            return False
        elif response.status_code == 403:
            print("\n❌ Token is valid but NO ACCESS to this competition")
            return False
        else:
            print(f"\n⚠️  Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("\n❌ Request timeout: Could not reach the API server")
        print("   → Check your internet connection")
        return False
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error: Could not reach the API server")
        print("   → Check your internet connection and firewall settings")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_sai_token()
