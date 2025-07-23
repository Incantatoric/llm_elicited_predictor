"""
Alpha Vantage News Collector for Hanwha Solutions

This module collects financial news from Alpha Vantage API with proper date filtering
and uses LLM to automatically score news impact for the Bayesian model.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import requests
from dotenv import load_dotenv
import openai

# Load environment variables from config/.env
load_dotenv("config/.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageNewsCollector:
    """Collects and scores financial news using Alpha Vantage API."""
    
    def __init__(self):
        """Initialize the news collector."""
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
        
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found - news scoring will be basic")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.data_dir = Path("data/news")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Alpha Vantage News Collector initialized")
    
    def get_news_for_month(self, year: int, month: int, company: str = "Hanwha Solutions") -> List[Dict]:
        """
        Get news articles for a specific month.
        
        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)
            company: Company name to search for
            
        Returns:
            List of news articles with metadata
        """
        logger.info(f"Collecting news for {company} in {year}-{month:02d}")
        
        # Calculate date range for the month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Alpha Vantage News API parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': '009830.KS',  # Hanwha Solutions ticker
            'apikey': self.api_key,
            'limit': 50,  # Maximum articles per request
            'time_from': start_date.strftime('%Y%m%dT0000'),
            'time_to': end_date.strftime('%Y%m%dT2359')
        }
        
        try:
            logger.info(f"Making API request for {start_date.date()} to {end_date.date()}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"API Error: {data['Error Message']}")
                return []
            
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                return []
            
            # Extract news articles
            articles = data.get('feed', [])
            logger.info(f"Found {len(articles)} articles for {year}-{month:02d}")
            
            # Process and filter articles
            processed_articles = []
            for article in articles:
                processed_article = self._process_article(article, year, month)
                if processed_article:
                    processed_articles.append(processed_article)
            
            return processed_articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def _process_article(self, article: Dict, year: int, month: int) -> Optional[Dict]:
        """Process a single news article."""
        try:
            # Extract publication date
            pub_date_str = article.get('time_published', '')
            if pub_date_str:
                pub_date = datetime.strptime(pub_date_str, '%Y%m%dT%H%M%S')
                # Verify it's in the correct month
                if pub_date.year != year or pub_date.month != month:
                    return None
            else:
                return None
            
            # Extract sentiment scores
            sentiment_data = article.get('ticker_sentiment', [])
            sentiment_score = 0.0
            if sentiment_data:
                # Find Hanwha Solutions sentiment
                for ticker_sentiment in sentiment_data:
                    if ticker_sentiment.get('ticker') == '009830.KS':
                        sentiment_score = float(ticker_sentiment.get('ticker_sentiment_score', 0))
                        break
            
            processed_article = {
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'published_date': pub_date.isoformat(),
                'sentiment_score': sentiment_score,
                'relevance_score': float(article.get('relevance_score', 0)),
                'ticker_sentiment_label': article.get('ticker_sentiment_label', ''),
                'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                'overall_sentiment_score': float(article.get('overall_sentiment_score', 0))
            }
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def score_news_impact(self, articles: List[Dict]) -> List[Dict]:
        """
        Score news impact using LLM or basic scoring.
        
        Args:
            articles: List of processed news articles
            
        Returns:
            List of articles with impact scores added
        """
        if not articles:
            return []
        
        if self.openai_api_key:
            return self._score_with_llm(articles)
        else:
            return self._score_basic(articles)
    
    def _score_with_llm(self, articles: List[Dict]) -> List[Dict]:
        """Score news impact using OpenAI GPT."""
        client = openai.OpenAI(api_key=self.openai_api_key)
        
        scored_articles = []
        
        for article in articles:
            try:
                # Create prompt for scoring
                prompt = f"""
                You are a financial analyst assessing the potential impact of news on Hanwha Solutions stock price.
                
                News Article:
                Title: {article['title']}
                Summary: {article['summary']}
                Source: {article['source']}
                Date: {article['published_date']}
                Sentiment Score: {article['sentiment_score']}
                
                Please provide:
                1. An impact score from 1-5 (1=minimal impact, 5=major impact)
                2. A brief reasoning for the score
                3. Whether this is likely to be positive, negative, or neutral for the stock
                
                Respond in JSON format:
                {{
                    "impact_score": <1-5>,
                    "reasoning": "<brief explanation>",
                    "direction": "<positive/negative/neutral>"
                }}
                """
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                # Parse response
                content = response.choices[0].message.content
                try:
                    score_data = json.loads(content)
                    article['llm_impact_score'] = score_data.get('impact_score', 3)
                    article['llm_reasoning'] = score_data.get('reasoning', '')
                    article['llm_direction'] = score_data.get('direction', 'neutral')
                except json.JSONDecodeError:
                    # Fallback to basic scoring
                    article['llm_impact_score'] = self._basic_impact_score(article)
                    article['llm_reasoning'] = 'LLM scoring failed, using basic scoring'
                    article['llm_direction'] = 'neutral'
                
            except Exception as e:
                logger.error(f"LLM scoring failed for article: {e}")
                article['llm_impact_score'] = self._basic_impact_score(article)
                article['llm_reasoning'] = 'LLM scoring failed, using basic scoring'
                article['llm_direction'] = 'neutral'
            
            scored_articles.append(article)
        
        return scored_articles
    
    def _score_basic(self, articles: List[Dict]) -> List[Dict]:
        """Basic scoring based on sentiment and relevance."""
        scored_articles = []
        
        for article in articles:
            impact_score = self._basic_impact_score(article)
            article['basic_impact_score'] = impact_score
            article['basic_reasoning'] = f"Based on sentiment ({article['sentiment_score']:.2f}) and relevance ({article['relevance_score']:.2f})"
            
            # Determine direction
            if article['sentiment_score'] > 0.1:
                direction = 'positive'
            elif article['sentiment_score'] < -0.1:
                direction = 'negative'
            else:
                direction = 'neutral'
            
            article['basic_direction'] = direction
            scored_articles.append(article)
        
        return scored_articles
    
    def _basic_impact_score(self, article: Dict) -> float:
        """Calculate basic impact score (1-5)."""
        # Base score from sentiment
        sentiment_score = abs(article.get('sentiment_score', 0))
        relevance_score = article.get('relevance_score', 0)
        
        # Combine scores
        impact_score = (sentiment_score * 2.5) + (relevance_score * 2.5)
        
        # Clamp to 1-5 range
        return max(1.0, min(5.0, impact_score))
    
    def collect_monthly_news(self, start_year: int, start_month: int, 
                           end_year: int, end_month: int) -> Dict:
        """
        Collect news for a range of months.
        
        Args:
            start_year, start_month: Start date
            end_year, end_month: End date
            
        Returns:
            Dictionary with monthly news data
        """
        all_news = {}
        
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        while current_date < end_date:
            year = current_date.year
            month = current_date.month
            
            logger.info(f"Processing {year}-{month:02d}")
            
            # Get news for this month
            articles = self.get_news_for_month(year, month)
            
            if articles:
                # Score the news
                scored_articles = self.score_news_impact(articles)
                
                # Calculate monthly news score
                monthly_score = self._calculate_monthly_score(scored_articles)
                
                all_news[f"{year}-{month:02d}"] = {
                    'articles': scored_articles,
                    'monthly_news_score': monthly_score,
                    'article_count': len(scored_articles)
                }
                
                logger.info(f"Collected {len(scored_articles)} articles for {year}-{month:02d}")
            else:
                logger.info(f"No articles found for {year}-{month:02d}")
                all_news[f"{year}-{month:02d}"] = {
                    'articles': [],
                    'monthly_news_score': 0.0,
                    'article_count': 0
                }
            
            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
        
        return all_news
    
    def _calculate_monthly_score(self, articles: List[Dict]) -> float:
        """Calculate overall monthly news impact score."""
        if not articles:
            return 0.0
        
        # Use LLM score if available, otherwise basic score
        scores = []
        for article in articles:
            if 'llm_impact_score' in article:
                scores.append(article['llm_impact_score'])
            elif 'basic_impact_score' in article:
                scores.append(article['basic_impact_score'])
        
        if scores:
            # Weight by relevance and take average
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for article in articles:
                relevance = article.get('relevance_score', 0.5)
                if 'llm_impact_score' in article:
                    score = article['llm_impact_score']
                else:
                    score = article.get('basic_impact_score', 3.0)
                
                total_weighted_score += score * relevance
                total_weight += relevance
            
            return total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def save_news_data(self, news_data: Dict, filename: str = None) -> str:
        """Save news data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alpha_vantage_news_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"News data saved to {filepath}")
        return str(filepath)
    
    def generate_monthly_scores_csv(self, news_data: Dict, output_file: str = None) -> str:
        """Generate CSV file with monthly news scores for model integration."""
        if output_file is None:
            output_file = "monthly_news_scores.csv"
        
        filepath = self.data_dir / output_file
        
        # Create CSV data
        csv_data = []
        for month_key, month_data in news_data.items():
            csv_data.append({
                'month': month_key,
                'news_score': month_data['monthly_news_score'],
                'article_count': month_data['article_count']
            })
        
        # Sort by month
        csv_data.sort(key=lambda x: x['month'])
        
        # Write CSV
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"Monthly scores CSV saved to {filepath}")
        return str(filepath)


def main():
    """Main function for testing the news collector."""
    try:
        collector = AlphaVantageNewsCollector()
        
        # Test with recent months (adjust as needed)
        print("Testing Alpha Vantage News Collector...")
        
        # Collect news for last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        news_data = collector.collect_monthly_news(
            start_date.year, start_date.month,
            end_date.year, end_date.month
        )
        
        # Save results
        json_file = collector.save_news_data(news_data)
        csv_file = collector.generate_monthly_scores_csv(news_data)
        
        print(f"✅ News collection completed!")
        print(f"📄 Detailed results: {json_file}")
        print(f"📊 Monthly scores: {csv_file}")
        
        # Print summary
        total_articles = sum(month_data['article_count'] for month_data in news_data.values())
        print(f"📰 Total articles collected: {total_articles}")
        
        for month, data in news_data.items():
            if data['article_count'] > 0:
                print(f"  {month}: {data['article_count']} articles, score: {data['monthly_news_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 