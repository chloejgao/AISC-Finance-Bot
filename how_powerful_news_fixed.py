#!/usr/bin/env python3
"""
📊 AI-Powered Market Impact Reporter
Automated News Event Analysis with Professional Reports

This script fetches market data, analyzes impact, and generates comprehensive
market intelligence reports in professional format.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from datetime import datetime, timedelta
import warnings
import requests
from typing import Dict, List, Optional, Tuple
import json
warnings.filterwarnings('ignore')

# Configuration for different event types and their typical market impacts
EVENT_CONFIGS = {
    'jobs_report': {
        'name': 'U.S. Jobs Report',
        'release_time': '08:30',
        'tickers': ['SPY', 'QQQ', 'XLF', 'TLT', 'DXY'],
        'typical_impact': 'High volatility in equities and bonds, USD strength on hot data',
        'sectors_to_watch': ['Financials (XLF)', 'Technology (QQQ)', 'Bonds (TLT)']
    },
    'cpi_report': {
        'name': 'U.S. CPI Inflation Report',
        'release_time': '08:30',
        'tickers': ['SPY', 'QQQ', 'TLT', 'GLD', 'DXY'],
        'typical_impact': 'Rate expectations shift, growth stocks sensitive to yields',
        'sectors_to_watch': ['Growth Tech (QQQ)', 'Treasuries (TLT)', 'Gold (GLD)']
    },
    'fomc_decision': {
        'name': 'Federal Reserve Policy Decision',
        'release_time': '14:00',
        'tickers': ['SPY', 'QQQ', 'XLF', 'TLT', 'VIX'],
        'typical_impact': 'Broad market reaction, yield curve movements, volatility spikes',
        'sectors_to_watch': ['Banks (XLF)', 'Bonds (TLT)', 'Volatility (VIX)']
    },
    'earnings': {
        'name': 'Major Earnings Announcement',
        'release_time': '16:00',
        'tickers': ['SPY', 'QQQ', 'Individual Stock'],
        'typical_impact': 'Sector rotation, individual stock volatility, sentiment shift',
        'sectors_to_watch': ['Sector ETFs', 'Individual Names', 'Options Activity']
    }
}

class MarketImpactReporter:
    def __init__(self, event_type: str = 'jobs_report', custom_config: Optional[Dict] = None):
        """Initialize the reporter with event configuration"""
        if custom_config:
            self.config = custom_config
        else:
            self.config = EVENT_CONFIGS.get(event_type, EVENT_CONFIGS['jobs_report'])
        
        self.results = {}
        self.market_data = {}
        
    def get_recent_trading_day(self, days_back: int = 1) -> str:
        """Get a recent trading day (avoiding weekends)"""
        today = datetime.now()
        
        while days_back <= 10:
            test_date = today - timedelta(days=days_back)
            
            # Skip weekends
            if test_date.weekday() < 5:  # Monday = 0, Friday = 4
                return test_date.strftime("%Y-%m-%d")
            
            days_back += 1
        
        # Fallback
        return (today - timedelta(days=3)).strftime("%Y-%m-%d")

    def fix_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix column naming issues from Yahoo Finance"""
        if data.empty:
            return data
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Create a mapping for common column name variations
        column_mapping = {}
        for col in data.columns:
            col_lower = str(col).lower().replace(' ', '').replace('_', '')
            
            if col_lower in ['adjclose', 'adj_close', 'adjustedclose']:
                column_mapping[col] = 'Adj Close'
            elif col_lower == 'close':
                column_mapping[col] = 'Close'
            elif col_lower == 'open':
                column_mapping[col] = 'Open'
            elif col_lower == 'high':
                column_mapping[col] = 'High'
            elif col_lower == 'low':
                column_mapping[col] = 'Low'
            elif col_lower == 'volume':
                column_mapping[col] = 'Volume'
        
        if column_mapping:
            data = data.rename(columns=column_mapping)
        
        # If we don't have 'Adj Close' but have 'Close', use Close as Adj Close
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        
        return data

    def fetch_market_data(self, ticker: str, event_date: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Fetch market data for analysis"""
        try:
            dt = datetime.strptime(event_date, "%Y-%m-%d")
            days_ago = (datetime.now().date() - dt.date()).days
            
            if days_ago <= 7:
                interval = "1m"
                start = dt - timedelta(days=1)
                end = dt + timedelta(days=1)
            elif days_ago <= 60:
                interval = "5m"
                start = dt - timedelta(days=2)
                end = dt + timedelta(days=2)
            else:
                interval = "1d"
                start = dt - timedelta(days=5)
                end = dt + timedelta(days=5)
            
            # Try to fetch data
            data = yf.download(ticker, 
                              start=start.strftime("%Y-%m-%d"), 
                              end=end.strftime("%Y-%m-%d"), 
                              interval=interval,
                              auto_adjust=False,
                              progress=False)
            
            if data.empty:
                return pd.DataFrame(), None
            
            data = self.fix_column_names(data)
            
            if 'Adj Close' not in data.columns:
                return pd.DataFrame(), None
            
            # Handle timezone for intraday data
            if interval in ["1m", "5m"]:
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    try:
                        data.index = data.index.tz_convert('US/Eastern').tz_localize(None)
                    except:
                        data.index = data.index.tz_localize(None)
                
                try:
                    data = data.between_time("09:30", "16:00")
                except:
                    pass
            
            # Calculate returns and volume changes
            if len(data) > 1:
                data['Returns'] = data['Adj Close'].pct_change()
                if 'Volume' in data.columns:
                    data['Volume_Change'] = data['Volume'].pct_change()
                else:
                    data['Volume'] = 0
                    data['Volume_Change'] = 0
            
            return data, interval
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame(), None

    def analyze_ticker_impact(self, ticker: str, market_data: pd.DataFrame, 
                            event_datetime: datetime, data_interval: str) -> Optional[Dict]:
        """Analyze impact for a single ticker"""
        try:
            if data_interval == "1d":
                return self.analyze_daily_impact(market_data, event_datetime.strftime("%Y-%m-%d"))
            else:
                event_day_data = market_data[market_data.index.date == event_datetime.date()]
                if event_day_data.empty:
                    return None
                
                return self.analyze_intraday_impact(event_day_data, event_datetime, 60, 60, data_interval)
        
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            return None

    def analyze_intraday_impact(self, event_day_data: pd.DataFrame, event_datetime: datetime, 
                              pre_window: int, post_window: int, data_interval: str) -> Optional[Dict]:
        """Perform intraday impact analysis"""
        if data_interval == "5m":
            pre_window = max(30, (pre_window // 5) * 5)
            post_window = max(30, (post_window // 5) * 5)
        
        pre_start = event_datetime - timedelta(minutes=pre_window)
        post_end = event_datetime + timedelta(minutes=post_window)
        
        pre_data = event_day_data[(event_day_data.index >= pre_start) & 
                                 (event_day_data.index < event_datetime)]
        post_data = event_day_data[(event_day_data.index >= event_datetime) & 
                                  (event_day_data.index < post_end)]
        
        if len(pre_data) < 3 or len(post_data) < 3:
            return None
        
        try:
            price_before = pre_data['Adj Close'].iloc[0]
            price_after = post_data['Adj Close'].iloc[-1]
            price_change = price_after - price_before
            price_change_pct = (price_change / price_before) * 100
            
            pre_volatility = pre_data['Returns'].std()
            post_volatility = post_data['Returns'].std()
            volatility_change = post_volatility - pre_volatility
            
            if 'Volume' in pre_data.columns and pre_data['Volume'].sum() > 0:
                pre_avg_volume = pre_data['Volume'].mean()
                post_avg_volume = post_data['Volume'].mean()
                volume_spike = post_avg_volume / pre_avg_volume if pre_avg_volume > 0 else 1
            else:
                volume_spike = 1
            
            # Statistical significance
            pre_returns = pre_data['Returns'].dropna()
            post_returns = post_data['Returns'].dropna()
            
            if len(pre_returns) > 1 and len(post_returns) > 1:
                try:
                    t_stat, p_value = ttest_ind(post_returns, pre_returns)
                    is_significant = p_value < 0.05
                except:
                    p_value = None
                    is_significant = False
            else:
                p_value = None
                is_significant = False
            
            return {
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volatility_change': volatility_change,
                'volume_spike': volume_spike,
                'p_value': p_value,
                'is_significant': is_significant,
                'pre_data': pre_data,
                'post_data': post_data,
                'price_before': price_before,
                'price_after': price_after
            }
            
        except Exception as e:
            return None

    def analyze_daily_impact(self, market_data: pd.DataFrame, event_date: str) -> Optional[Dict]:
        """Perform daily impact analysis"""
        try:
            dt = datetime.strptime(event_date, "%Y-%m-%d")
            event_day_data = market_data[market_data.index.date == dt.date()]
            
            if event_day_data.empty:
                return None
            
            event_return = event_day_data['Returns'].iloc[0] if not event_day_data['Returns'].isna().all() else 0
            event_volume = event_day_data['Volume'].iloc[0] if 'Volume' in event_day_data.columns else 0
            
            other_days = market_data[market_data.index.date != dt.date()]
            if not other_days.empty and not other_days['Returns'].isna().all():
                avg_return = other_days['Returns'].mean()
                if 'Volume' in other_days.columns and other_days['Volume'].sum() > 0:
                    avg_volume = other_days['Volume'].mean()
                    volume_ratio = event_volume / avg_volume if avg_volume > 0 else 1
                else:
                    volume_ratio = 1
            else:
                avg_return = 0
                volume_ratio = 1
            
            return {
                'event_return': event_return,
                'avg_return': avg_return,
                'volume_ratio': volume_ratio,
                'event_day_data': event_day_data,
                'price_change_pct': event_return * 100
            }
        except Exception as e:
            return None

    def generate_market_narrative(self, ticker: str, results: Dict) -> str:
        """Generate narrative explanation for market moves"""
        narratives = []
        
        if 'price_change_pct' in results:
            change_pct = results['price_change_pct']
            
            if abs(change_pct) > 2.0:
                direction = "surged" if change_pct > 0 else "plummeted" 
                narratives.append(f"{ticker} {direction} {abs(change_pct):.1f}% following the announcement")
            elif abs(change_pct) > 0.5:
                direction = "rose" if change_pct > 0 else "fell"
                narratives.append(f"{ticker} {direction} {abs(change_pct):.1f}%")
            else:
                narratives.append(f"{ticker} showed minimal reaction ({change_pct:+.1f}%)")
        
        if 'volume_spike' in results and results['volume_spike'] > 1.5:
            narratives.append(f"trading volume spiked {results['volume_spike']:.1f}x above normal")
        elif 'volume_ratio' in results and results['volume_ratio'] > 1.5:
            narratives.append(f"volume was {results['volume_ratio']:.1f}x higher than average")
        
        if 'volatility_change' in results and results['volatility_change'] > 0.001:
            narratives.append("volatility increased significantly")
        
        return "; ".join(narratives) if narratives else f"{ticker} showed limited reaction"

    def generate_professional_report(self, event_date: str, event_description: str = None) -> str:
        """Generate a professional market impact report"""
        
        report_lines = []
        report_lines.append("**⭐️ Market Impact Analysis Report**")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Header
        event_name = event_description or f"{self.config['name']} - {event_date}"
        report_lines.append(f"**📊 Event: {event_name}**")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("**1. Executive Summary**")
        report_lines.append("")
        
        if not self.results:
            report_lines.append("❌ No market data available for analysis.")
            return "\n".join(report_lines)
        
        # Generate summary based on strongest reactions
        summary_items = []
        strongest_movers = []
        
        for ticker, result in self.results.items():
            if result and 'price_change_pct' in result:
                change_pct = result['price_change_pct']
                if abs(change_pct) > 0.3:  # Only include significant moves
                    strongest_movers.append((ticker, change_pct))
        
        strongest_movers.sort(key=lambda x: abs(x[1]), reverse=True)
        
        if strongest_movers:
            top_mover = strongest_movers[0]
            direction = "positive" if top_mover[1] > 0 else "negative"
            summary_items.append(f"Primary market reaction was {direction}, with {top_mover[0]} showing the strongest response ({top_mover[1]:+.1f}%)")
        
        # Add volume and volatility insights
        high_volume_tickers = [ticker for ticker, result in self.results.items() 
                              if result and (result.get('volume_spike', 1) > 1.5 or result.get('volume_ratio', 1) > 1.5)]
        
        if high_volume_tickers:
            summary_items.append(f"Elevated trading activity observed across {', '.join(high_volume_tickers)}")
        
        report_lines.extend(summary_items)
        report_lines.append("")
        
        # Market Reaction Details
        report_lines.append("**2. Market Reaction Details**")
        report_lines.append("")
        
        for ticker in self.config['tickers']:
            if ticker in self.results and self.results[ticker]:
                narrative = self.generate_market_narrative(ticker, self.results[ticker])
                report_lines.append(f"• **{ticker}**: {narrative}")
        
        report_lines.append("")
        
        # Analysis and Interpretation
        report_lines.append("**3. Why This Happened**")
        report_lines.append("")
        
        # Generate explanations based on patterns
        explanations = []
        
        # Check for broad market moves
        spy_result = self.results.get('SPY')
        if spy_result and abs(spy_result.get('price_change_pct', 0)) > 0.5:
            change = spy_result['price_change_pct']
            if change > 0:
                explanations.append("1. **Risk-on sentiment dominated** as investors interpreted the data as supportive of continued economic growth or favorable policy conditions.")
            else:
                explanations.append("1. **Risk-off sentiment prevailed** as the data raised concerns about economic headwinds or policy tightening.")
        
        # Check for yield sensitivity
        tlt_result = self.results.get('TLT')
        qqq_result = self.results.get('QQQ')
        if tlt_result and qqq_result:
            tlt_change = tlt_result.get('price_change_pct', 0)
            qqq_change = qqq_result.get('price_change_pct', 0)
            
            if tlt_change * qqq_change < 0 and abs(tlt_change) > 0.3:  # Opposite moves
                if tlt_change < 0:  # Bonds down = yields up
                    explanations.append("2. **Interest rate expectations shifted higher**, pressuring bond prices and benefiting rate-sensitive sectors while challenging growth stocks.")
                else:  # Bonds up = yields down
                    explanations.append("2. **Interest rate expectations shifted lower**, supporting bond prices and growth assets as discount rates declined.")
        
        # Check for financial sector reaction
        xlf_result = self.results.get('XLF')
        if xlf_result and abs(xlf_result.get('price_change_pct', 0)) > 0.5:
            change = xlf_result['price_change_pct']
            if change > 0:
                explanations.append("3. **Financial sector strength** suggests market expectations of rising rates or improved lending conditions.")
            else:
                explanations.append("3. **Financial sector weakness** indicates concerns about credit conditions or falling rate expectations.")
        
        if not explanations:
            explanations.append("Market reaction was muted, suggesting the data was largely in line with expectations or other offsetting factors were at play.")
        
        report_lines.extend(explanations)
        report_lines.append("")
        
        # Trading Suggestions
        report_lines.append("**4. Strategic Implications**")
        report_lines.append("")
        report_lines.append("| **Time Frame** | **Strategy** | **Rationale** |")
        report_lines.append("|---|---|---|")
        
        # Generate suggestions based on market moves
        if strongest_movers:
            top_ticker, top_change = strongest_movers[0]
            
            if abs(top_change) > 1.0:
                if top_change > 0:
                    report_lines.append(f"| Short-term (1-2 weeks) | Monitor {top_ticker} for continuation vs. reversal | Strong initial reaction may lead to follow-through or profit-taking |")
                else:
                    report_lines.append(f"| Short-term (1-2 weeks) | Look for {top_ticker} oversold bounce opportunities | Sharp selloff may create tactical entry points |")
            
            report_lines.append("| Medium-term (1-3 months) | Reassess positioning based on follow-up data | Initial market reaction needs confirmation from subsequent releases |")
            report_lines.append("| Risk Management | Set stops below key technical levels | Market volatility may persist as new information is processed |")
        else:
            report_lines.append("| All timeframes | Maintain current positioning | Limited market reaction suggests no major regime change |")
            report_lines.append("| Monitor | Watch for confirming signals in next data releases | Muted reaction may precede larger moves |")
        
        report_lines.append("")
        
        # Key Takeaway
        report_lines.append("**5. Key Takeaway**")
        report_lines.append("")
        
        if strongest_movers and abs(strongest_movers[0][1]) > 1.0:
            report_lines.append(f"The {event_name} generated significant market volatility, with clear directional moves across asset classes. This suggests the data materially shifted investor expectations and warrants close monitoring of follow-up developments.")
        else:
            report_lines.append(f"The {event_name} produced a measured market response, indicating the data was largely anticipated. Future positioning should remain flexible as additional information becomes available.")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Generated by AI-Powered Market Impact Analyzer*")
        report_lines.append(f"*Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(report_lines)

    def run_full_analysis(self, event_date: str = None, event_description: str = None) -> str:
        """Run complete analysis and generate report"""
        
        if not event_date:
            event_date = self.get_recent_trading_day()
        
        print(f"🚀 Analyzing Market Impact for {self.config['name']}")
        print(f"📅 Event Date: {event_date}")
        print(f"🕐 Event Time: {self.config['release_time']}")
        print("=" * 60)
        
        # Create event datetime
        dt = datetime.strptime(event_date, "%Y-%m-%d")
        event_time = datetime.strptime(self.config['release_time'], "%H:%M").time()
        event_datetime = datetime.combine(dt.date(), event_time)
        
        # Analyze each ticker
        for ticker in self.config['tickers']:
            print(f"📊 Analyzing {ticker}...")
            
            market_data, data_interval = self.fetch_market_data(ticker, event_date)
            
            if not market_data.empty:
                self.market_data[ticker] = market_data
                result = self.analyze_ticker_impact(ticker, market_data, event_datetime, data_interval)
                self.results[ticker] = result
                
                if result:
                    change_pct = result.get('price_change_pct', 0)
                    print(f"   ✅ {ticker}: {change_pct:+.2f}%")
                else:
                    print(f"   ⚠️ {ticker}: Analysis incomplete")
            else:
                print(f"   ❌ {ticker}: No data available")
                self.results[ticker] = None
        
        print("\n" + "=" * 60)
        print("📋 Generating Professional Report...")
        print("=" * 60)
        
        # Generate and return the report
        report = self.generate_professional_report(event_date, event_description)
        print(report)
        
        return report

def main():
    """Main function with example usage"""
    
    # Example 1: Jobs Report Analysis
    print("Example 1: Jobs Report Analysis")
    jobs_reporter = MarketImpactReporter('jobs_report')
    jobs_report = jobs_reporter.run_full_analysis(
        event_description="December 2024 U.S. Employment Report - Hot Labor Market Data"
    )
    
    print("\n" + "="*80 + "\n")
    
    # Example 2: CPI Report Analysis  
    print("Example 2: CPI Report Analysis")
    cpi_reporter = MarketImpactReporter('cpi_report')
    cpi_report = cpi_reporter.run_full_analysis(
        event_description="January 2025 U.S. CPI Report - Inflation Trends"
    )
    
    print("\n" + "="*80 + "\n")
    
    # Example 3: Custom Event Analysis
    print("Example 3: Custom Event Analysis")
    custom_config = {
        'name': 'Tech Earnings Surprise',
        'release_time': '16:00',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'QQQ', 'SPY'],
        'typical_impact': 'Sector rotation and individual stock volatility',
        'sectors_to_watch': ['Technology', 'Growth Stocks', 'Options']
    }
    
    custom_reporter = MarketImpactReporter(custom_config=custom_config)
    custom_report = custom_reporter.run_full_analysis(
        event_description="Major Tech Earnings Beat Expectations"
    )

if __name__ == "__main__":
    main()