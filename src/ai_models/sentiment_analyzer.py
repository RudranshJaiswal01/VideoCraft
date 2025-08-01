from transformers import pipeline
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment in text content for film editing context.
    Provides detailed sentiment analysis beyond basic positive/negative classification.
    """
    
    def __init__(self, config: dict):
        """
        Initialize SentimentAnalyzer with pre-trained model.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        
        # Load sentiment analysis model
        try:
            model_name = config['models']['sentiment_analyzer']
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                return_all_scores=True
            )
            logger.info(f"Sentiment analyzer loaded: {model_name}")
        except Exception as e:
            logger.error(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.sentiment_analyzer or not text.strip():
            return {'neutral': 1.0}
        
        try:
            results = self.sentiment_analyzer(text)
            
            if isinstance(results[0], list):
                results = results[0]
            
            # Convert to consistent format
            sentiment_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Map labels to consistent format
                if 'pos' in label or 'positive' in label:
                    sentiment_scores['positive'] = score
                elif 'neg' in label or 'negative' in label:
                    sentiment_scores['negative'] = score  
                else:
                    sentiment_scores['neutral'] = score
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'neutral': 1.0}
    
    def analyze_dialogue_sentiment_timeline(self, dialogue_data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment across dialogue timeline.
        
        Args:
            dialogue_data: List of dialogue entries
            
        Returns:
            Timeline of sentiment analysis
        """
        sentiment_timeline = []
        
        for item in dialogue_data:
            text = item.get('text', '')
            sentiment = self.analyze_sentiment(text)
            
            timeline_entry = {
                'timestamp': item.get('estimated_start_time', 0),
                'line_number': item.get('line_number', 0),
                'speaker': item.get('speaker', 'Unknown'),
                'text': text,
                'sentiment_scores': sentiment,
                'dominant_sentiment': max(sentiment, key=sentiment.get),
                'sentiment_confidence': max(sentiment.values())
            }
            
            sentiment_timeline.append(timeline_entry)
        
        return sentiment_timeline
    
    def detect_sentiment_shifts(self, sentiment_timeline: List[Dict], threshold: float = 0.4) -> List[Dict]:
        """
        Detect significant sentiment shifts in timeline.
        
        Args:
            sentiment_timeline: Timeline of sentiment analysis
            threshold: Minimum change for shift detection
            
        Returns:
            List of sentiment shift points
        """
        shifts = []
        
        for i in range(1, len(sentiment_timeline)):
            prev_sentiment = sentiment_timeline[i-1]['sentiment_scores']
            curr_sentiment = sentiment_timeline[i]['sentiment_scores']
            
            # Calculate sentiment change
            change = self._calculate_sentiment_change(prev_sentiment, curr_sentiment)
            
            if change > threshold:
                shift = {
                    'timestamp': sentiment_timeline[i]['timestamp'],
                    'from_sentiment': sentiment_timeline[i-1]['dominant_sentiment'],
                    'to_sentiment': sentiment_timeline[i]['dominant_sentiment'],
                    'change_magnitude': change,
                    'context': sentiment_timeline[i]['text'][:100] + '...'
                }
                shifts.append(shift)
        
        return shifts
    
    def _calculate_sentiment_change(self, sent1: Dict[str, float], sent2: Dict[str, float]) -> float:
        """Calculate magnitude of sentiment change between two points."""
        all_sentiments = set(sent1.keys()) | set(sent2.keys())
        
        change = 0.0
        for sentiment in all_sentiments:
            val1 = sent1.get(sentiment, 0.0)
            val2 = sent2.get(sentiment, 0.0)
            change += abs(val1 - val2)
        
        return change / len(all_sentiments) if all_sentiments else 0.0
