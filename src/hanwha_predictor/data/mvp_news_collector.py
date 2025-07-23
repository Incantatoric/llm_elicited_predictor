"""
MVP News Collector for Hanwha Solutions

Quick and dirty news collection for MVP - gets news from 2022-2025
and extracts items with links for manual verification.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MVPNewsCollector:
    """Quick news collector for MVP."""
    
    def __init__(self):
        """Initialize the collector."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.data_dir = Path("data/news")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("MVP News Collector initialized")
    
    def collect_hanwha_news(self) -> Dict:
        """
        Collect Hanwha Solutions news from 2022-2025.
        Focus on news that would impact stock price.
        """
        logger.info("Starting Hanwha news collection...")
        
        # Single comprehensive search
        search_prompt = """
        Search for major news about Hanwha Solutions (한화솔루션) from 2022-2025 that would impact stock price.
        
        Please find news articles that would significantly affect the company's stock returns.
        Focus on: earnings announcements, acquisitions, partnerships, regulatory changes, major investments, 
        solar energy developments, hydrogen projects, and other significant business events.
        
        For each news article you find, please provide:
        1. Article title
        2. Publication date
        3. Source/website
        4. URL
        5. Brief summary of the news and its potential impact on stock price
        
        Please search thoroughly and provide as many relevant news items as possible.
        """
        
        logger.info("Making API request...")
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": search_prompt
                }],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 8
                }]
            )
            
            logger.info("Received response from Anthropic API")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response content length: {len(response.content) if response.content else 0}")
            
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
            logger.error(f"Error during search: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _extract_news_from_response(self, response, period: str) -> List[Dict]:
        """Extract news items from Anthropic response."""
        news_items = []
        
        # Process each content block
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == 'text':
                text = content_block.text
                
                # Look for news items in the text
                items = self._parse_news_from_text(text, period)
                news_items.extend(items)
        
        return news_items
    
    def _parse_news_from_text(self, text: str, period: str) -> List[Dict]:
        """Parse news items from text response."""
        news_items = []
        
        # Split by potential news separators
        lines = text.split('\n')
        current_item = {}
        
        for line in lines:
            line = line.strip()
            
            # Look for patterns that indicate news items
            if any(keyword in line.lower() for keyword in ['title:', 'headline:', 'news:', 'article:']):
                # Save previous item if exists
                if current_item:
                    current_item['period'] = period
                    news_items.append(current_item)
                
                # Start new item
                current_item = {
                    'title': line.split(':', 1)[1].strip() if ':' in line else line,
                    'date': '',
                    'source': '',
                    'url': '',
                    'summary': '',
                    'period': period
                }
            
            elif 'date:' in line.lower() or 'published:' in line.lower():
                if current_item:
                    current_item['date'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif 'source:' in line.lower() or 'url:' in line.lower():
                if current_item:
                    current_item['source'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif 'summary:' in line.lower() or 'description:' in line.lower():
                if current_item:
                    current_item['summary'] = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif line and current_item and not any(keyword in line.lower() for keyword in ['title:', 'date:', 'source:', 'summary:']):
                # This might be additional summary text
                if current_item.get('summary'):
                    current_item['summary'] += ' ' + line
                else:
                    current_item['summary'] = line
        
        # Add the last item
        if current_item:
            current_item['period'] = period
            news_items.append(current_item)
        
        return news_items
    
    def save_news_data(self, news_data: Dict) -> str:
        """Save news data to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mvp_hanwha_news_{timestamp}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"News data saved to {filepath}")
        return str(filepath)
    
    def generate_verification_csv(self, news_data: Dict) -> str:
        """Generate CSV for manual verification."""
        import csv
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hanwha_news_verification_{timestamp}.csv"
        filepath = self.data_dir / filename
        
        # Flatten news data
        all_items = []
        for period, items in news_data.items():
            for item in items:
                item['search_period'] = period
                all_items.append(item)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if all_items:
                fieldnames = ['search_period', 'title', 'date', 'source', 'url', 'summary']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in all_items:
                    row = {
                        'search_period': item.get('search_period', ''),
                        'title': item.get('title', ''),
                        'date': item.get('date', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'summary': item.get('summary', '')
                    }
                    writer.writerow(row)
        
        logger.info(f"Verification CSV saved to {filepath}")
        return str(filepath)

    def save_raw_response(self, search_result: Dict) -> str:
        """Save the raw API response for manual review."""
        logger.info("Saving raw response...")
        
        if not search_result.get("success"):
            logger.error("Cannot save response - search failed")
            return None
            
        # Create directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.data_dir / f"hanwha_news_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        
        # Save the full response for reference
        full_response_file = session_dir / "full_search_response.txt"
        
        try:
            with open(full_response_file, 'w', encoding='utf-8') as f:
                f.write(f"Search Query: Hanwha Solutions news 2022-2025\n")
                f.write(f"Search Date: {datetime.now().isoformat()}\n")
                f.write(f"Number of searches performed: {search_result.get('search_count', 'unknown')}\n")
                f.write("="*80 + "\n\n")
                
                response_content = search_result["response"].content
                logger.info(f"Processing {len(response_content)} content blocks...")
                
                for i, content_block in enumerate(response_content):
                    logger.info(f"Processing content block {i}...")
                    logger.info(f"  Block type: {type(content_block)}")
                    
                    try:
                        # Handle different types of content blocks
                        if hasattr(content_block, 'type'):
                            f.write(f"\n--- Content Block {i} (Type: {content_block.type}) ---\n")
                            
                            if content_block.type == 'text':
                                if hasattr(content_block, 'text') and content_block.text is not None:
                                    f.write(content_block.text)
                                    logger.info(f"  Wrote text content (length: {len(content_block.text)})")
                                else:
                                    logger.warning(f"  Text block has no text content")
                                    f.write("[No text content available]")
                                    
                            elif content_block.type == 'server_tool_use':
                                f.write(f"Tool Use: {getattr(content_block, 'name', 'Unknown')}\n")
                                if hasattr(content_block, 'input'):
                                    f.write(f"Input: {json.dumps(content_block.input, indent=2, ensure_ascii=False)}\n")
                                logger.info(f"  Processed server_tool_use block")
                                
                            elif content_block.type == 'web_search_tool_result':
                                f.write("Web Search Results:\n")
                                if hasattr(content_block, 'content'):
                                    f.write(f"{json.dumps(content_block.content, indent=2, ensure_ascii=False)}\n")
                                logger.info(f"  Processed web_search_tool_result block")
                                
                            else:
                                # Handle unknown block types
                                f.write(f"Unknown block type: {content_block.type}\n")
                                f.write(f"Raw content: {str(content_block)}\n")
                                logger.warning(f"  Unknown block type: {content_block.type}")
                        else:
                            # Fallback for blocks without type attribute
                            f.write(f"\n--- Content Block {i} (No type attribute) ---\n")
                            if hasattr(content_block, 'text') and content_block.text is not None:
                                f.write(content_block.text)
                                logger.info(f"  Wrote text from block without type")
                            else:
                                f.write(f"Raw content: {str(content_block)}\n")
                                logger.warning(f"  Block has no text attribute or text is None")
                        
                        f.write("\n\n")
                        
                    except Exception as block_error:
                        logger.error(f"Error processing content block {i}: {block_error}", exc_info=True)
                        f.write(f"[Error processing this block: {block_error}]\n\n")
                        
        except Exception as file_error:
            logger.error(f"Error writing to file: {file_error}", exc_info=True)
            return None
        
        # Save metadata
        metadata = {
            "query": "Hanwha Solutions news 2022-2025",
            "search_date": datetime.now().isoformat(),
            "search_count": search_result.get('search_count', 0),
            "session_folder": str(session_dir),
            "content_blocks_count": len(response_content)
        }
        
        metadata_file = session_dir / "metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
        
        logger.info(f"Raw response saved successfully to {session_dir}")
        return str(session_dir)


def main():
    """Main function."""
    print("🚀 MVP Hanwha News Collector")
    print("=" * 50)
    
    try:
        collector = MVPNewsCollector()
        
        # Collect news
        print("📰 Collecting Hanwha Solutions news...")
        search_result = collector.collect_hanwha_news()
        
        if search_result.get("success"):
            # Save raw response
            session_dir = collector.save_raw_response(search_result)
            
            if session_dir:
                print(f"\n✅ Collection completed!")
                print(f"📁 Raw response saved to: {session_dir}")
                print(f"📄 Full response: {session_dir}/full_search_response.txt")
                print(f"📊 Metadata: {session_dir}/metadata.json")
                print(f"🔍 Searches performed: {search_result.get('search_count', 'unknown')}")
                
                print(f"\n🔍 Next steps:")
                print(f"1. Open {session_dir}/full_search_response.txt")
                print(f"2. Review the raw API response")
                print(f"3. Extract news items manually from the text")
                print(f"4. Verify dates to avoid look-ahead bias")
                print(f"5. We'll integrate the verified data into the model")
            else:
                print("\n❌ Failed to save response")
        else:
            print(f"\n❌ Search failed: {search_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 