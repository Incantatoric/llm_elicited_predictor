#!/usr/bin/env python3
"""
News collection module for Hanwha Solutions stock prediction.
Uses Tavily API to collect relevant news for each month.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env")

logger = logging.getLogger(__name__)

class HanwhaNewsCollector:
    """Collects news data for Hanwha Solutions using Tavily API."""
    
    def __init__(self):
        """Initialize the news collector."""
        # Force reload from the correct .env file
        load_dotenv("config/.env", override=True)
        
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        #print(f"Using TAVILY_API_KEY: {self.api_key}")
        
        # Initialize Tavily client
        self.client = TavilyClient(self.api_key)
        self.news_dir = Path("data/news")
        self.news_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Hanwha News Collector")
    
    def collect_monthly_news(self, year: int, month: int, max_results: int = 5) -> List[Dict]:
        """
        Collect news for a specific month.
        
        Args:
            year: Year to collect news for
            month: Month to collect news for (1-12)
            max_results: Maximum number of news items to return
            
        Returns:
            List of news dictionaries with date, headline, summary, source, url
        """
        # Create date range for the month
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - date.resolution
        else:
            end_date = date(year, month + 1, 1) - date.resolution
        
        # Search queries for Hanwha Solutions
        search_queries = [
            "Hanwha Solutions",
            "Hanwha Solutions stock",
            "Hanwha Solutions earnings",
            "Hanwha Solutions news",
            "Hanwha Q Cells",  # Subsidiary
            "Hanwha Solutions solar",
            "Hanwha Solutions chemicals"
        ]
        
        all_news = []
        
        for query in search_queries:
            try:
                news_items = self._search_tavily(
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_results
                )
                # Add query information to each news item
                for item in news_items:
                    item['query'] = query
                all_news.extend(news_items)
                logger.info(f"Found {len(news_items)} news items for query: {query}")
                
            except Exception as e:
                logger.warning(f"Error searching for query '{query}': {e}")
                continue
        
        # Remove duplicates and sort by date
        unique_news = self._deduplicate_news(all_news)
        unique_news.sort(key=lambda x: x.get('date', ''))
        
        # Limit to max_results
        return unique_news[:max_results]
    
    def _search_tavily(self, query: str, start_date: date, end_date: date, max_results: int) -> List[Dict]:
        """Search Tavily API for news using the Python client."""
        try:
            # Add date context to query
            date_query = f"{query} {start_date.strftime('%B %Y')}"
            
            response = self.client.search(
                query=date_query,
                search_depth="basic",
                max_results=max_results,
                include_domains=[
                    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
                    "koreajoongangdaily.joins.com", "koreatimes.co.kr", "koreaherald.com",
                    "money2.daishin.com", "yna.co.kr", "yonhapnews.co.kr"
                ]
            )
            
            results = response.get('results', [])
            
            # Process results into our format
            news_items = []
            for result in results:
                try:
                    news_item = {
                        'date': result.get('published_date', ''),
                        'headline': result.get('title', ''),
                        'summary': result.get('content', '')[:500],  # Limit summary length
                        'source': result.get('source', ''),
                        'url': result.get('url', ''),
                        'query': query  # Track which query found this
                    }
                    news_items.append(news_item)
                except Exception as e:
                    logger.warning(f"Error processing result: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Tavily API error: {e}")
            return []
    
    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on URL, preserving query information."""
        url_to_item = {}
        
        for item in news_items:
            url = item.get('url', '')
            if url:
                if url not in url_to_item:
                    url_to_item[url] = item
                else:
                    # If same URL found by multiple queries, combine query info
                    existing_item = url_to_item[url]
                    existing_queries = existing_item.get('query', '').split('; ')
                    new_query = item.get('query', '')
                    if new_query not in existing_queries:
                        existing_queries.append(new_query)
                        existing_item['query'] = '; '.join(existing_queries)
        
        return list(url_to_item.values())
    
    def save_monthly_news(self, year: int, month: int, news_items: List[Dict]):
        """Save news items to JSON file for a specific month."""
        filename = self.news_dir / f"news_{year}_{month:02d}.json"
        
        data = {
            'year': year,
            'month': month,
            'collected_date': datetime.now().isoformat(),
            'news_count': len(news_items),
            'news_items': news_items
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(news_items)} news items to {filename}")
    
    def collect_all_months(self, start_year: int = 2022, end_year: int = 2025):
        """Collect news for all months in the date range."""
        logger.info(f"Starting news collection for {start_year}-{end_year}")
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months
                if year == 2025 and month > 6:
                    break
                
                try:
                    logger.info(f"Collecting news for {year}-{month:02d}")
                    news_items = self.collect_monthly_news(year, month)
                    self.save_monthly_news(year, month, news_items)
                    
                    # Be nice to the API
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error collecting news for {year}-{month:02d}: {e}")
                    continue


def main():
    """Test the news collector."""
    logging.basicConfig(level=logging.INFO)
    
    collector = HanwhaNewsCollector()
    
    # Test with a few months
    test_months = [
        (2025, 4),  # The April 2025 example
        (2024, 12),
        (2023, 6)
    ]
    
    for year, month in test_months:
        print(f"\nCollecting news for {year}-{month:02d}...")
        news_items = collector.collect_monthly_news(year, month)
        collector.save_monthly_news(year, month, news_items)
        
        print(f"Found {len(news_items)} news items:")
        for i, item in enumerate(news_items, 1):
            print(f"{i}. {item['date']} - {item['headline'][:80]}...")
            print(f"   Source: {item['source']}")
            print()


if __name__ == "__main__":
    main() 