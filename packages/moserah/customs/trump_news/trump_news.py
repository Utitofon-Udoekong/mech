import json
from typing import Optional, Dict, Any, Tuple, Callable, List
import os
import re
import datetime
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from newsapi import NewsApiClient
import openai
from enum import Enum
from collections import Counter

# Type Definitions
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

class ReportingTone(str, Enum):
    SERIOUS = "serious"
    SATIRICAL = "satirical"
    LATE_NIGHT = "late_night"

class APIClients:
    def __init__(self, api_keys: Any):
        self.news_api_key = api_keys.get("news_api")
        self.openai_api_key = api_keys.get("openai")
        
        if not self.news_api_key:
            raise ValueError("Missing NEWS_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
            
        # Set up API clients
        self.newsapi = NewsApiClient(api_key=self.news_api_key)
        openai.api_key = self.openai_api_key
        
        # Set up NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('sentiment/vader_lexicon.zip')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('vader_lexicon')
            nltk.download('stopwords')
        
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                if "openai" in str(e).lower() and retries_left["openai"] > 0:
                    retries_left["openai"] -= 1
                    api_keys.rotate("openai")
                    return execute()
                elif "newsapi" in str(e).lower() and retries_left["news_api"] > 0:
                    retries_left["news_api"] -= 1
                    api_keys.rotate("news_api")
                    return execute()
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper

def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None

@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the Trump news analysis task.
    
    Args:
        prompt: The user's query about Trump news
        api_keys: Dictionary containing API keys
        **kwargs: Additional arguments including:
            - tone: Reporting tone (serious, satirical, late_night)
            - timeframe: Time window for news (1d, 7d, 30d)
            - source_filter: Optional list of news sources
    
    Returns:
        Tuple containing:
        - response: The news analysis response
        - prompt: The original prompt
        - metadata: Additional metadata about the analysis
        - None: Reserved for future use
    """
    try:
        clients = APIClients(api_keys)
        
        # Parse parameters
        tone = kwargs.get("tone", "serious")
        timeframe = kwargs.get("timeframe", "1d")
        source_filter = kwargs.get("source_filter", None)
        
        # Parse and validate inputs
        query, specific_topic = _parse_prompt(prompt)
        
        # Get raw news data
        articles = _fetch_news(clients, query, timeframe, source_filter)
        if not articles:
            return error_response("No relevant news found for the given parameters.")
        
        # Analyze the news
        analysis = _analyze_news(clients, articles, specific_topic)
        
        # Generate the report
        report = _generate_report(clients, analysis, ReportingTone(tone))
        
        # Prepare metadata
        metadata_dict = {
            "query": query,
            "specific_topic": specific_topic,
            "timeframe": timeframe,
            "tone": tone,
            "article_count": len(articles),
            "sources": list(set([article["source"]["name"] for article in articles])),
            "sentiment_analysis": {
                "average_sentiment": analysis["average_sentiment"],
                "sentiment_distribution": analysis["sentiment_distribution"]
            },
            "key_topics": analysis["key_topics"][:5]
        }
        
        return report, prompt, metadata_dict, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e), "", None, None

def _parse_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    """Parse the user prompt to extract the query and any specific topic focus."""
    query = "Trump"
    specific_topic = None
    
    topics = [
        "economy", "immigration", "foreign policy", "legal", "campaign", 
        "social media", "rally", "election", "poll", "speech"
    ]
    
    for topic in topics:
        if re.search(r'\b' + topic + r'\b', prompt.lower()):
            specific_topic = topic
            query = f"Trump {topic}"
            break
    
    return query, specific_topic

def _fetch_news(clients: APIClients, query: str, timeframe: str, source_filter: Optional[List[str]] = None) -> List[Dict]:
    """Fetch news articles from the News API."""
    today = datetime.datetime.now()
    if timeframe == "1d":
        from_date = today - datetime.timedelta(days=1)
    elif timeframe == "7d":
        from_date = today - datetime.timedelta(days=7)
    else:
        from_date = today - datetime.timedelta(days=30)
    
    from_date_str = from_date.strftime('%Y-%m-%d')
    
    try:
        params = {
            'q': query,
            'from_param': from_date_str,
            'language': 'en',
            'sort_by': 'publishedAt'
        }
        
        if source_filter:
            params['sources'] = ','.join(source_filter)
        
        response = clients.newsapi.get_everything(**params)
        return response.get('articles', [])
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return []

def _analyze_news(clients: APIClients, articles: List[Dict], specific_topic: Optional[str]) -> Dict[str, Any]:
    """Analyze the news articles to extract key insights."""
    analysis = {
        "articles": articles,
        "sources": [article["source"]["name"] for article in articles],
        "sentiment_scores": [],
        "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
        "average_sentiment": 0,
        "key_topics": [],
        "fact_check_articles": [],
        "article_summaries": []
    }
    
    all_text = ""
    
    for article in articles:
        text = article["title"]
        if article.get("description"):
            text += " " + article["description"]
        
        sentiment = clients.sia.polarity_scores(text)
        analysis["sentiment_scores"].append(sentiment)
        
        compound = sentiment["compound"]
        if compound >= 0.05:
            analysis["sentiment_distribution"]["positive"] += 1
        elif compound <= -0.05:
            analysis["sentiment_distribution"]["negative"] += 1
        else:
            analysis["sentiment_distribution"]["neutral"] += 1
        
        all_text += " " + text
        
        summary = _summarize_article(article)
        analysis["article_summaries"].append({
            "title": article["title"],
            "source": article["source"]["name"],
            "published_at": article["publishedAt"],
            "summary": summary,
            "sentiment": sentiment["compound"]
        })
    
    if analysis["sentiment_scores"]:
        compounds = [score["compound"] for score in analysis["sentiment_scores"]]
        analysis["average_sentiment"] = sum(compounds) / len(compounds)
    
    analysis["key_topics"] = _extract_key_topics(clients, all_text)
    
    return analysis

def _extract_key_topics(clients: APIClients, text: str) -> List[str]:
    """Extract key topics from the combined text of all articles."""
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in clients.stop_words]
    
    word_counts = Counter(words)
    common_words = word_counts.most_common(15)
    
    return [word for word, _ in common_words]

def _summarize_article(article: Dict[str, Any]) -> str:
    """Generate a brief summary of an article."""
    summary = article["title"]
    
    if article.get("description"):
        sentences = sent_tokenize(article["description"])
        if sentences:
            summary += " " + sentences[0]
    
    return summary

def _generate_report(clients: APIClients, analysis: Dict[str, Any], tone: ReportingTone) -> str:
    """Generate a report based on the analysis and desired tone."""
    system_prompts = {
        ReportingTone.SERIOUS: (
            "You are a balanced, professional political news analyst. "
            "Provide factual, objective reporting on the given news about Donald Trump. "
            "Avoid both positive and negative bias. Cite specific sources and present multiple perspectives. "
            "Focus on accuracy and context."
        ),
        ReportingTone.SATIRICAL: (
            "You are a satirical political commentator in the style of The Onion or The Daily Show. "
            "Provide witty, ironic commentary on the given news about Donald Trump. "
            "Use humor to highlight absurdities and contradictions, but maintain a foundation of truth. "
            "Employ exaggeration and parody while still conveying the essence of the news."
        ),
        ReportingTone.LATE_NIGHT: (
            "You are a late-night comedy show host like Stephen Colbert or Jimmy Kimmel. "
            "Provide humorous, punchy commentary on the given news about Donald Trump. "
            "Use punchlines, zinger jokes, and amusing analogies. "
            "Be playfully irreverent while still conveying the core facts."
        )
    }
    
    sources_str = "\n".join([f"- {article['source']['name']}: {article['title']}" for article in analysis["articles"][:5]])
    
    summaries = []
    for i, summary in enumerate(analysis["article_summaries"][:5]):
        summaries.append(f"Article {i+1}: {summary['title']} ({summary['source']})")
        summaries.append(f"Summary: {summary['summary']}")
        sentiment_text = "positive" if summary["sentiment"] > 0.05 else "negative" if summary["sentiment"] < -0.05 else "neutral"
        summaries.append(f"Sentiment: {sentiment_text}")
        summaries.append("")
    
    summaries_str = "\n".join(summaries)
    
    user_prompt = f"""
    Generate a news report about Donald Trump based on the following information:
    
    Key topics: {', '.join(analysis['key_topics'][:5])}
    
    Overall sentiment: {analysis['average_sentiment']:.2f} 
    (Sentiment distribution: {analysis['sentiment_distribution']['positive']} positive, 
    {analysis['sentiment_distribution']['neutral']} neutral, 
    {analysis['sentiment_distribution']['negative']} negative)
    
    Top sources and headlines:
    {sources_str}
    
    Article summaries:
    {summaries_str}
    
    Based on this information, generate a concise news report about Donald Trump.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompts[tone]
            },
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7 if tone == ReportingTone.SERIOUS else 0.9,
        max_tokens=800
    )
    
    return response.choices[0].message.content.strip()
