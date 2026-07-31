import httpx
from datetime import datetime, timedelta
from langchain_core.tools import tool
from config import DataAPIConfig

# Persistent HTTP client for speed (keeps connections alive)
client = httpx.Client(timeout=5.0)


@tool
def get_asset_news_and_price(asset_query: str) -> str:
    """
    Fetches current price and latest structured news for an asset.
    Input: Ticker symbol (e.g., 'O:GOLD' for Gold ETF, 'FX:EURUSD' for Euro/Dollar, 'BTCUSD' for Crypto).
    """
    try:
        # 1. Get Price (Using Finnhub quote)
        quote_url = f"https://finnhub.io/api/v1/quote?symbol={asset_query}&token={DataAPIConfig.FINNHUB_API_KEY}"
        quote_resp = client.get(quote_url).json()

        if 'c' in quote_resp and quote_resp['c'] != 0:
            price_info = f"Current Price: {quote_resp['c']} | Daily Change: {quote_resp['d']:.2f} ({quote_resp['dp']:.2f}%)"
        else:
            # Fallback for Forex pairs if Finnhub quote fails, try a generic search or return N/A
            price_info = "Live price data temporarily unavailable for this specific ticker format."

        # 2. Get News (Using Finnhub general market/forex news)
        today = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

        # Finnhub news endpoint (using 'forex' or 'general' category)
        news_url = f"https://finnhub.io/api/v1/news?category=forex&token={DataAPIConfig.FINNHUB_API_KEY}"
        news_resp = client.get(news_url).json()

        headlines = []
        if news_resp and isinstance(news_resp, list):
            for article in news_resp[:4]:  # Top 4 to keep it brief for voice
                headline = article.get('headline', 'No Title')
                source = article.get('source', 'Unknown')
                summary = article.get('summary', '')[:120]  # Truncate for voice brevity
                headlines.append(f"- [{source}] {headline}. ({summary})")

        news_str = "\n".join(headlines) if headlines else "No recent news available."

        return f"Asset: {asset_query}\n{price_info}\n\nLatest News:\n{news_str}"

    except Exception as e:
        return f"CRITICAL ERROR fetching data for {asset_query}: {str(e)}"


@tool
def get_economic_calendar(currency_code: str) -> str:
    """
    Fetches today's high-impact economic calendar events for a specific currency.
    Input: 3-letter currency code (e.g., 'USD', 'EUR', 'GBP', 'JPY').
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        # Finnhub calendar requires from and to dates
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

        # 🚀 FINNHUB ECONOMIC CALENDAR ENDPOINT
        cal_url = f"https://finnhub.io/api/v1/calendar/economic?from={from_date}&to={to_date}&token={DataAPIConfig.FINNHUB_API_KEY}"
        cal_resp = client.get(cal_url).json()

        if 'economicCalendar' not in cal_resp or not cal_resp['economicCalendar']:
            return f"No economic calendar data available for {currency_code} at this time."

        events = cal_resp['economicCalendar']

        # Filter for the requested currency AND high importance (3 = High Impact)
        # We also check if the currency is in the country/event name as a fallback
        filtered_events = []
        for event in events:
            event_country = event.get('country', '').upper()
            event_importance = event.get('importance', 0)

            # Match currency to country (e.g., USD -> US, EUR -> EU, GBP -> UK/GB)
            country_map = {'USD': ['US', 'USA'], 'EUR': ['EU', 'EZ'], 'GBP': ['UK', 'GB'], 'JPY': ['JP'], 'AUD': ['AU'],
                           'CAD': ['CA']}
            valid_countries = country_map.get(currency_code, [currency_code])

            if event_country in valid_countries and event_importance >= 2:  # 2=Medium, 3=High
                event_name = event.get('event', 'Unknown Event')
                event_date = event.get('date', 'Unknown Date')
                actual = event.get('actual', 'Pending')
                forecast = event.get('estimate', 'N/A')
                previous = event.get('previous', 'N/A')

                # Format the time nicely (Finnhub returns ISO format)
                try:
                    dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                    time_str = dt.strftime('%b %d, %H:%M')
                except:
                    time_str = event_date

                filtered_events.append(
                    f"• {event_name} | Time: {time_str} | Actual: {actual} | Forecast: {forecast} | Previous: {previous}"
                )

        if not filtered_events:
            return f"No medium or high-impact economic events scheduled for {currency_code} in the next 48 hours."

        # Limit to top 3 events to prevent the LLM from reading a massive list
        top_events = filtered_events[:3]
        return f"Upcoming High-Impact Events for {currency_code}:\n" + "\n".join(top_events)

    except Exception as e:
        return f"Error fetching calendar: {str(e)}"


FINANCIAL_TOOLS = [get_asset_news_and_price, get_economic_calendar]