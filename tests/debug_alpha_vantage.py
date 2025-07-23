"""
Alpha Vantage API Diagnostic Script

This script tests the Alpha Vantage API step by step to understand:
1. If the API key works
2. What ticker symbols are supported
3. How date filtering works
4. What the raw responses look like
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env")

def test_api_key():
    """Test if the API key is valid."""
    print("=== Testing API Key ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Test with a simple request
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'AAPL',
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'Error Message' in data:
                print(f"❌ API Error: {data['Error Message']}")
                return False
            elif 'Note' in data:
                print(f"⚠️ API Note: {data['Note']}")
                return False
            else:
                print("✅ API key is valid")
                return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_news_api_basic():
    """Test basic news API functionality."""
    print("\n=== Testing News API (Basic) ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = "https://www.alphavantage.co/query"
    
    # Test 1: Basic news request without date filtering
    print("Test 1: Basic news request for AAPL")
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': 'AAPL',
        'apikey': api_key,
        'limit': 5
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            
            if 'feed' in data:
                articles = data['feed']
                print(f"Found {len(articles)} articles")
                
                if articles:
                    print("Sample article structure:")
                    sample = articles[0]
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print("No articles found")
            else:
                print("No 'feed' key in response")
                print(f"Full response: {json.dumps(data, indent=2)}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_hanwha_tickers():
    """Test different ticker symbols for Hanwha."""
    print("\n=== Testing Hanwha Ticker Symbols ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = "https://www.alphavantage.co/query"
    
    # Test different ticker symbols
    tickers_to_test = [
        '009830.KS',  # Korean ticker
        'HANWHA',     # Generic search
        '009830',     # Just the number
        'HWS',        # Possible abbreviation
        'HWSOL'       # Another possible abbreviation
    ]
    
    for ticker in tickers_to_test:
        print(f"\nTesting ticker: {ticker}")
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': api_key,
            'limit': 3
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    articles = data['feed']
                    print(f"  Found {len(articles)} articles")
                    
                    if articles:
                        # Show first article title
                        first_article = articles[0]
                        title = first_article.get('title', 'No title')
                        print(f"  First article: {title[:80]}...")
                    else:
                        print("  No articles found")
                else:
                    print(f"  No 'feed' key. Response keys: {list(data.keys())}")
                    if 'Error Message' in data:
                        print(f"  Error: {data['Error Message']}")
                    elif 'Note' in data:
                        print(f"  Note: {data['Note']}")
            else:
                print(f"  HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"  Request failed: {e}")

def test_date_filtering():
    """Test date filtering functionality."""
    print("\n=== Testing Date Filtering ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = "https://www.alphavantage.co/query"
    
    # Test with recent dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Testing date range: {start_date.date()} to {end_date.date()}")
    
    # Test with AAPL first (known to work)
    print("\nTest 1: AAPL with date filtering")
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': 'AAPL',
        'apikey': api_key,
        'limit': 5,
        'time_from': start_date.strftime('%Y%m%dT0000'),
        'time_to': end_date.strftime('%Y%m%dT2359')
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'feed' in data:
                articles = data['feed']
                print(f"Found {len(articles)} articles with date filtering")
                
                if articles:
                    print("Sample articles with dates:")
                    for i, article in enumerate(articles[:3]):
                        title = article.get('title', 'No title')
                        pub_date = article.get('time_published', 'No date')
                        print(f"  {i+1}. {title[:60]}... | {pub_date}")
                else:
                    print("No articles found with date filtering")
            else:
                print("No 'feed' key in response")
        else:
            print(f"HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test with Hanwha ticker
    print("\nTest 2: Hanwha with date filtering")
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': '009830.KS',
        'apikey': api_key,
        'limit': 5,
        'time_from': start_date.strftime('%Y%m%dT0000'),
        'time_to': end_date.strftime('%Y%m%dT2359')
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            if 'feed' in data:
                articles = data['feed']
                print(f"Found {len(articles)} Hanwha articles with date filtering")
                
                if articles:
                    print("Sample Hanwha articles:")
                    for i, article in enumerate(articles[:3]):
                        title = article.get('title', 'No title')
                        pub_date = article.get('time_published', 'No date')
                        print(f"  {i+1}. {title[:60]}... | {pub_date}")
                else:
                    print("No Hanwha articles found with date filtering")
            else:
                print("No 'feed' key in response")
        else:
            print(f"HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def test_broader_search():
    """Test broader search terms."""
    print("\n=== Testing Broader Search ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = "https://www.alphavantage.co/query"
    
    # Test with broader search terms
    search_terms = [
        'Korea',      # Country
        'Korean',     # Language
        'Asia',       # Region
        'Energy',     # Sector
        'Solar'       # Industry
    ]
    
    for term in search_terms:
        print(f"\nTesting search term: {term}")
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'topics': term,
            'apikey': api_key,
            'limit': 3
        }
        
        try:
            response = requests.get(url, params=params)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'feed' in data:
                    articles = data['feed']
                    print(f"  Found {len(articles)} articles")
                    
                    if articles:
                        # Check if any mention Hanwha
                        hanwha_mentions = []
                        for article in articles:
                            title = article.get('title', '').lower()
                            summary = article.get('summary', '').lower()
                            if 'hanwha' in title or 'hanwha' in summary:
                                hanwha_mentions.append(article.get('title', 'No title'))
                        
                        if hanwha_mentions:
                            print(f"  Found {len(hanwha_mentions)} articles mentioning Hanwha:")
                            for mention in hanwha_mentions:
                                print(f"    - {mention[:60]}...")
                        else:
                            print("  No Hanwha mentions found")
                    else:
                        print("  No articles found")
                else:
                    print(f"  No 'feed' key. Response keys: {list(data.keys())}")
            else:
                print(f"  HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"  Request failed: {e}")

def main():
    """Run all diagnostic tests."""
    print("🔍 Alpha Vantage API Diagnostic")
    print("=" * 50)
    
    # Test 1: API key
    if not test_api_key():
        print("\n❌ API key test failed. Stopping.")
        return
    
    # Test 2: Basic news API
    test_news_api_basic()
    
    # Test 3: Hanwha tickers
    test_hanwha_tickers()
    
    # Test 4: Date filtering
    test_date_filtering()
    
    # Test 5: Broader search
    test_broader_search()
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic complete!")
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Check if any ticker symbols work")
    print("3. See if broader searches find Hanwha-related news")
    print("4. Determine if date filtering is working properly")

if __name__ == "__main__":
    main() 