#!/usr/bin/env python3
"""
Test script to demonstrate date-filtered news collection capabilities.
"""

import os
import requests
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv("config/.env")

def test_yahoo_finance_api():
    """Test Yahoo Finance API with date filtering."""
    print("=== Testing Yahoo Finance API ===")
    
    # Yahoo Finance API endpoint
    url = "https://yfapi.net/v8/finance/chart/009830.KS"
    
    # Parameters for April 2025
    params = {
        'period1': int(datetime(2025, 4, 1).timestamp()),
        'period2': int(datetime(2025, 4, 30).timestamp()),
        'interval': '1d'
    }
    
    headers = {
        'x-api-key': 'YOUR_YAHOO_API_KEY'  # Would need actual API key
    }
    
    print(f"Would search for news from {params['period1']} to {params['period2']}")
    print("Note: Yahoo Finance API requires paid subscription")
    print()

def test_newsapi():
    """Test NewsAPI.org with date filtering."""
    print("=== Testing NewsAPI.org ===")
    
    api_key = os.getenv('NEWSAPI_KEY')  # Would need to add this
    if not api_key:
        print("NEWSAPI_KEY not found - would need to add to .env")
        return
    
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': 'Hanwha Solutions',
        'from': '2025-04-01',
        'to': '2025-04-30',
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': api_key
    }
    
    print(f"Would search: {params}")
    print("NewsAPI.org has good date filtering but limited free tier")
    print()

def test_alpha_vantage():
    """Test Alpha Vantage News API."""
    print("=== Testing Alpha Vantage News API ===")
    
    api_key = os.getenv('ALPHA_VANTAGE_KEY')  # Would need to add this
    if not api_key:
        print("ALPHA_VANTAGE_KEY not found - would need to add to .env")
        return
    
    url = "https://www.alphavantage.co/query"
    
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': '009830.KS',
        'time_from': '20250401T0000',
        'time_to': '20250430T2359',
        'apikey': api_key
    }
    
    print(f"Would search: {params}")
    print("Alpha Vantage has good financial news filtering")
    print()

def test_anthropic_raw():
    """Test Anthropic web search without domain restrictions."""
    print("=== Testing Anthropic Web Search (Raw) ===")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ANTHROPIC_API_KEY not found")
        return
    
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = """
    Search for news about Hanwha Solutions (009830.KS) that was published specifically in April 2025.
    
    Please find news articles that were actually published during April 2025, not news about events that happened in April 2025.
    
    For each news item, provide:
    1. The exact publication date
    2. Headline
    3. Brief summary
    4. Source URL
    
    Only include news that was published in April 2025.
    """
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2
            }]
        )
        
        print("Raw Anthropic response:")
        for block in response.content:
            if block.type == "text":
                print(block.text)
                print("-" * 50)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing date-filtered news collection capabilities...\n")
    
    test_yahoo_finance_api()
    test_newsapi()
    test_alpha_vantage()
    test_anthropic_raw()
    
    print("\n=== SUMMARY ===")
    print("1. Yahoo Finance API: Best for financial news, requires paid subscription")
    print("2. NewsAPI.org: Good date filtering, limited free tier")
    print("3. Alpha Vantage: Financial news specific, good date filtering")
    print("4. Anthropic: Intelligent but no guaranteed date filtering")
    print("\nRecommendation: Try Alpha Vantage or Yahoo Finance API for reliable date filtering.") 