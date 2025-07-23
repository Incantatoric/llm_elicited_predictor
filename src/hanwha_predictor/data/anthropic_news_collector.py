#!/usr/bin/env python3
"""
News collection module for Hanwha Solutions stock prediction using Anthropic web search.
Uses Claude's web search to find impactful news for each month.
"""

import os
import json
import logging
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env")

logger = logging.getLogger(__name__)

class AnthropicNewsCollector:
    """Collects news data for Hanwha Solutions using Anthropic web search."""
    
    def __init__(self):
        """Initialize the news collector."""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"
        
        self.news_dir = Path("data/news")
        self.news_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Anthropic News Collector")
    
    def collect_monthly_news(self, year: int, month: int, max_results: int = 5) -> List[Dict]:
        """
        Collect impactful news for a specific month using Claude's web search.
        
        Args:
            year: Year to collect news for
            month: Month to collect news for (1-12)
            max_results: Maximum number of news items to return
            
        Returns:
            List of news dictionaries with date, headline, summary, source, url
        """
        month_name = date(year, month, 1).strftime("%B %Y")
        
        # Create a focused prompt for impactful news
        prompt = f"""
        I need to find the most impactful news events that affected Hanwha Solutions (009830.KS) stock price in {month_name}.
        
        Please search for and identify the top {max_results} most significant news events that would have had the biggest impact on Hanwha Solutions stock returns during {month_name}.
        
        Focus on:
        1. Earnings announcements and financial results
        2. Major business developments (new divisions, acquisitions, partnerships)
        3. Regulatory decisions affecting the company
        4. Significant contracts or investments
        5. Management changes or strategic announcements
        6. Industry-specific news (solar, chemicals, construction) that directly impacts Hanwha Solutions
        
        For each news event, provide:
        - Publication date (if available)
        - Headline
        - Brief summary of the news and its potential impact
        - Source URL
        - Why this news would be impactful for stock price
        
        Exclude routine announcements, minor partnerships, or news about Hanwha Group affiliates that don't directly affect Hanwha Solutions.
        
        Only include news that was published during {month_name} or very close to it.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                    "allowed_domains": [
                        "bloomberg.com",
                        "koreajoongangdaily.joins.com", 
                        "koreatimes.co.kr", 
                        "koreaherald.com",
                        "money2.daishin.com",
                        "finance.yahoo.com", 
                        "investing.com"
                    ]
                }]
            )
            
            # Parse the response to extract news items
            news_items = self._parse_claude_response(response, month_name)
            
            logger.info(f"Found {len(news_items)} impactful news items for {month_name}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error collecting news for {month_name}: {e}")
            return []
    
    def _parse_claude_response(self, response, month_name: str) -> List[Dict]:
        """Parse Claude's response to extract structured news data."""
        news_items = []
        
        # Extract content from response
        content = response.content
        
        # Look for structured information in the response
        # Claude should provide news items in a structured format
        text_content = ""
        for block in content:
            if block.type == "text":
                text_content += block.text
        
        # Try to extract news items from the text
        # This is a simplified parser - we might need to improve it
        lines = text_content.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for patterns that indicate news items
            if line.startswith(('1.', '2.', '3.', '4.', '5.')) or 'Headline:' in line:
                if current_item:
                    news_items.append(current_item)
                current_item = {'month': month_name}
                
            elif 'Headline:' in line:
                current_item['headline'] = line.split('Headline:', 1)[1].strip()
            elif 'Date:' in line:
                current_item['date'] = line.split('Date:', 1)[1].strip()
            elif 'Summary:' in line:
                current_item['summary'] = line.split('Summary:', 1)[1].strip()
            elif 'Source:' in line or 'URL:' in line:
                current_item['source'] = line.split(':', 1)[1].strip()
            elif 'Impact:' in line:
                current_item['impact'] = line.split('Impact:', 1)[1].strip()
        
        # Add the last item
        if current_item:
            news_items.append(current_item)
        
        # If parsing failed, create a simple structure
        if not news_items:
            news_items = [{
                'month': month_name,
                'headline': 'News collected via Claude web search',
                'summary': text_content[:500],
                'date': '',
                'source': 'Anthropic web search',
                'raw_response': text_content
            }]
        
        return news_items
    
    def save_monthly_news(self, year: int, month: int, news_items: List[Dict]):
        """Save news items to JSON file for a specific month."""
        filename = self.news_dir / f"anthropic_news_{year}_{month:02d}.json"
        
        data = {
            'year': year,
            'month': month,
            'collected_date': datetime.now().isoformat(),
            'news_count': len(news_items),
            'news_items': news_items,
            'method': 'anthropic_web_search'
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(news_items)} news items to {filename}")
    
    def collect_all_months(self, start_year: int = 2022, end_year: int = 2025):
        """Collect news for all months in the date range."""
        logger.info(f"Starting Anthropic news collection for {start_year}-{end_year}")
        
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
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting news for {year}-{month:02d}: {e}")
                    continue


def main():
    """Test the Anthropic news collector."""
    logging.basicConfig(level=logging.INFO)
    
    collector = AnthropicNewsCollector()
    
    # Test with a few months
    test_months = [
        (2025, 4),  # The April 2025 example
        (2024, 12),
        (2023, 6)
    ]
    
    for year, month in test_months:
        print(f"\nCollecting impactful news for {year}-{month:02d}...")
        news_items = collector.collect_monthly_news(year, month)
        collector.save_monthly_news(year, month, news_items)
        
        print(f"Found {len(news_items)} impactful news items:")
        for i, item in enumerate(news_items, 1):
            print(f"{i}. {item.get('headline', 'No headline')}")
            print(f"   Summary: {item.get('summary', 'No summary')[:100]}...")
            print(f"   Source: {item.get('source', 'No source')}")
            print()


if __name__ == "__main__":
    main() 