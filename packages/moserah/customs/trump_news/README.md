# Trump News Mech Tool

A sophisticated AI-powered tool for tracking, analyzing, and reporting on news related to Donald Trump. This tool leverages advanced NLP techniques and multiple AI models to provide comprehensive news coverage with customizable reporting styles.

## Features

### 1. News Aggregation & Filtering
- Real-time tracking of Trump-related stories from multiple sources
- Intelligent filtering for relevance and bias
- Support for custom source filtering
- Time-based news aggregation (1d, 7d, 30d)

### 2. Sentiment Analysis & Topic Extraction
- Advanced sentiment analysis using NLTK's VADER
- Automatic topic extraction from news articles
- Sentiment distribution tracking across sources
- Key topic identification and trending themes

### 3. Multi-Tone Reporting
The tool supports three distinct reporting tones:
- **Serious**: Balanced, professional political news analysis
- **Satirical**: Witty, ironic commentary in the style of The Onion
- **Late Night**: Humorous, punchy commentary like late-night shows

### 4. Comprehensive Analysis
- Article summarization
- Source credibility tracking
- Sentiment distribution analysis
- Key topic identification
- Cross-source verification

## Usage

### Basic Usage
```python
result = run(
    prompt="What's the latest on Trump's legal issues?",
    api_keys={
        "news_api": "your_news_api_key",
        "openai": "your_openai_api_key"
    }
)
```

### Advanced Usage
```python
result = run(
    prompt="Show me Trump's latest campaign developments",
    api_keys={
        "news_api": "your_news_api_key",
        "openai": "your_openai_api_key"
    },
    tone="satirical",  # Options: "serious", "satirical", "late_night"
    timeframe="7d",    # Options: "1d", "7d", "30d"
    source_filter=["reuters", "ap", "bbc"]  # Optional list of sources
)
```

### Response Format
The tool returns a tuple containing:
1. **response**: The generated news report
2. **prompt**: The original query
3. **metadata**: Additional analysis data including:
   - Query and specific topic
   - Timeframe and tone used
   - Article count and sources
   - Sentiment analysis results
   - Key topics identified

## Dependencies
- nltk >= 3.8.1
- newsapi-python >= 0.2.7
- openai >= 1.0.0
- python-dotenv >= 1.0.0
- requests >= 2.31.0

## API Requirements
- News API key for news article fetching
- OpenAI API key for report generation

## Example Output
```json
{
    "response": "Generated news report...",
    "metadata": {
        "query": "Trump legal issues",
        "specific_topic": "legal",
        "timeframe": "7d",
        "tone": "serious",
        "article_count": 15,
        "sources": ["Reuters", "AP", "BBC"],
        "sentiment_analysis": {
            "average_sentiment": -0.2,
            "sentiment_distribution": {
                "positive": 3,
                "neutral": 7,
                "negative": 5
            }
        },
        "key_topics": ["court", "indictment", "trial", "appeal", "judge"]
    }
}
```

## Error Handling
The tool includes robust error handling for:
- API failures
- Invalid inputs
- Missing API keys
- Network issues
- Rate limiting

## Best Practices
1. Always provide valid API keys
2. Use specific prompts for better results
3. Consider using source filters for focused analysis
4. Monitor API usage and rate limits
5. Handle responses appropriately based on tone

## Limitations
- News API has rate limits and coverage restrictions
- OpenAI API costs may vary based on usage
- Some sources may require additional authentication
- Sentiment analysis accuracy depends on article quality

## Future Enhancements
- Integration with additional news sources
- Enhanced fact-checking capabilities
- Media bias analysis
- Narrative tracking across time
- Meme and soundbite generation
- Custom topic tracking
- Advanced filtering options 