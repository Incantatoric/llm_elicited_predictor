#!/usr/bin/env python3
"""
MVP Negative News Collector for Hanwha Solutions
================================================

This module collects negative news, challenges, risks, and problems affecting Hanwha Solutions
stock price. It focuses on:
- Supply chain disruptions
- Regulatory challenges  
- Competition from Chinese manufacturers
- Project delays and cost overruns
- Environmental/ESG controversies
- Management issues
- Market share losses
- Legal/regulatory investigations
- Analyst downgrades
- Financial challenges

Usage:
    python src/hanwha_predictor/data/mvp_negative_news_collector.py
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MVPNegativeNewsCollector:
    """Collects negative news and challenges affecting Hanwha Solutions."""
    
    def __init__(self):
        """Initialize the negative news collector."""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.data_dir = Path("data/news")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_hanwha_negative_news(self) -> Dict[str, Any]:
        """
        Collect negative news and challenges affecting Hanwha Solutions.
        
        Returns:
            Dict containing the search results and metadata
        """
        logger.info("Starting Hanwha negative news collection...")
        
        prompt = """You are a financial analyst specializing in identifying risks, challenges, and negative developments that could impact stock prices.

Your task is to search for and collect negative news, challenges, risks, and problems affecting Hanwha Solutions (KRX: 009830) from 2022-2025 that could negatively impact their stock price.

Focus specifically on:
1. **Supply Chain Issues**: Disruptions, shortages, delays, cost increases
2. **Regulatory Challenges**: Policy changes, compliance issues, legal investigations
3. **Competition**: Chinese manufacturer competition, market share losses, price wars
4. **Project Problems**: Delays, cost overruns, cancellations, technical issues
5. **Financial Challenges**: Debt issues, cash flow problems, credit downgrades
6. **Management Issues**: Executive departures, internal conflicts, strategic missteps
7. **Environmental/ESG Problems**: Controversies, violations, sustainability challenges
8. **Market Headwinds**: Industry downturns, demand weakness, economic factors
9. **Analyst Downgrades**: Negative ratings, target price cuts, bearish reports
10. **Legal Issues**: Lawsuits, investigations, regulatory fines

Search for news that shows:
- Problems with REC Silicon acquisition or operations
- Challenges with US expansion plans
- Supply chain disruptions affecting solar manufacturing
- Competition from Chinese solar manufacturers
- Regulatory hurdles in key markets
- Project delays or cancellations
- Financial difficulties or debt issues
- Management turnover or strategic problems
- Environmental controversies
- Legal or compliance issues

For each significant negative development, provide:
- Date of the news
- Source and headline
- Brief description of the negative impact
- Potential stock price implications

Focus on material negative developments that would realistically concern investors and potentially drive the stock price down.

Search thoroughly and be comprehensive in identifying risks and challenges."""

        try:
            logger.info("Making API request...")
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 8
                }]
            )
            
            logger.info("Received response from Anthropic API")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response content length: {len(response.content)}")
            
            # Log response structure for debugging
            logger.info("Response structure analysis:")
            for i, content_block in enumerate(response.content):
                logger.info(f"  Content block {i}:")
                if hasattr(content_block, 'type'):
                    logger.info(f"    Block type: {content_block.type}")
                if hasattr(content_block, 'text'):
                    logger.info(f"    Text length: {len(content_block.text) if content_block.text else 'None'}")
                    if content_block.text:
                        logger.info(f"    Text preview: {content_block.text[:200]}...")
            
            # Log usage information
            if hasattr(response, 'usage'):
                logger.info(f"Token usage: {response.usage}")
                if hasattr(response.usage, 'server_tool_use'):
                    logger.info(f"Server tool use: {response.usage.server_tool_use}")
            
            return {
                "success": True,
                "response": response,
                "search_count": response.usage.server_tool_use.web_search_requests if hasattr(response.usage, 'server_tool_use') else 0
            }
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """
        Save the negative news collection results to a file.
        
        Args:
            results: The search results dictionary
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hanwha_negative_news_{timestamp}.json"
        filepath = self.data_dir / filename
        
        # Extract the full response text
        if results.get("success") and hasattr(results["response"], "content"):
            full_response = ""
            for content_block in results["response"].content:
                if hasattr(content_block, 'text') and content_block.text:
                    full_response += content_block.text + "\n\n"
            
            # Create the data structure
            data = {
                "timestamp": timestamp,
                "search_metadata": {
                    "success": results["success"],
                    "search_count": results.get("search_count", 0),
                    "error": results.get("error", None)
                },
                "full_response": full_response,
                "negative_news_focus": [
                    "Supply chain disruptions and delays",
                    "Regulatory challenges and compliance issues", 
                    "Competition from Chinese manufacturers",
                    "Project delays and cost overruns",
                    "Financial challenges and debt issues",
                    "Management problems and turnover",
                    "Environmental/ESG controversies",
                    "Market headwinds and demand weakness",
                    "Analyst downgrades and bearish reports",
                    "Legal issues and investigations"
                ]
            }
        else:
            data = {
                "timestamp": timestamp,
                "search_metadata": results,
                "full_response": "Search failed",
                "negative_news_focus": []
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)

def main():
    """Main function to run the negative news collector."""
    print("🚀 MVP Hanwha Negative News Collector")
    print("=" * 50)
    
    collector = MVPNegativeNewsCollector()
    logger.info("MVP Negative News Collector initialized")
    
    print("📰 Collecting Hanwha Solutions negative news and challenges...")
    results = collector.collect_hanwha_negative_news()
    
    if results["success"]:
        print("✅ Search completed successfully")
        filepath = collector.save_results(results)
        print(f"📁 Results saved to: {filepath}")
        
        # Show a preview of the results
        if hasattr(results["response"], "content"):
            print("\n📋 Preview of negative news found:")
            print("-" * 40)
            for i, content_block in enumerate(results["response"].content):
                if hasattr(content_block, 'text') and content_block.text:
                    # Show first 500 characters of each block
                    preview = content_block.text[:500]
                    if len(content_block.text) > 500:
                        preview += "..."
                    print(f"Block {i+1}: {preview}\n")
    else:
        print(f"❌ Search failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 