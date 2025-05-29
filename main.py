import os
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_lookup, pitching_stats, batting_stats, statcast_batter
import requests
import python_weather
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from sklearn.linear_model import LinearRegression
import redis
import json
import multiprocessing as mp
from functools import partial
import sqlite3
import time
import logging
from contextlib import contextmanager
from tqdm.auto import tqdm
import concurrent.futures
from io import StringIO
import argparse
import sys
import pytz
import random
import statsapi
import aiohttp
from datetime import datetime, timedelta, timezone
import logging
import redis
from contextlib import contextmanager
import pybaseball
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup, schedule_and_record, team_batting, team_pitching
import warnings
from joblib import Parallel, delayed
from typing import Dict, Any, Optional
from math import sqrt

# Import Retrosheet integration layer
try:
    from retrosheet_integration_layer import create_retrosheet_integration_layer
    RETROSHEET_INTEGRATION_AVAILABLE = True
except ImportError:
    RETROSHEET_INTEGRATION_AVAILABLE = False
    logging.warning("Retrosheet integration layer not available - will use MLB Stats API only")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLBHitTracker')

class ProgressParallel:
    """Wrapper for parallel processing with progress bar"""
    def __init__(self, total, desc="", **kwargs):
        self.total = total
        self.desc = desc
        self.kwargs = kwargs

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with tqdm(total=self.total, desc=self.desc, **self.kwargs) as pbar:
                def update(*args):
                    pbar.update()

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = []
                    for arg in args[0]:  # args[0] is the iterable
                        future = executor.submit(func, arg)
                        future.add_done_callback(lambda p: update())
                        futures.append(future)
                        
                    results = []
                    for future in concurrent.futures.as_completed(futures):
                        results.append(future.result())
                    
                    return results
        return wrapper

@contextmanager
def timer(operation_name):
    """Context manager for timing operations"""
    start_time = time.time()
    yield
    duration = time.time() - start_time
    logger.info(f"{operation_name} took {duration:.2f} seconds")

# Redis setup
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_EXPIRATION = 60 * 60 * 24  # 24 hours in seconds

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# Database setup
def setup_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('mlb_stats.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS park_factors (
            venue_id INTEGER PRIMARY KEY,
            venue_name TEXT,
            factor REAL,
            last_updated TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def handle_api_failure(response, api_name):
    """Handle API failure by printing response and exiting"""
    print(f"\nAPI Failure in {api_name}")
    print(f"Status Code: {response.status_code}")
    print("\nResponse Headers:")
    for key, value in response.headers.items():
        print(f"{key}: {value}")
    print("\nResponse Body:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    sys.exit(1)

class MLBStatsAPI:
    """MLB Stats API client"""
    
    BASE_URL = "https://statsapi.mlb.com/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_schedule_by_date(self, date):
        """Get schedule for a specific date with enhanced probable pitcher extraction"""
        try:
            # Use hydrated endpoint to get probable pitcher information
            url = f"{self.BASE_URL}/schedule?sportId=1&date={date}&hydrate=team,linescore,flags,liveLookin,review,broadcasts,decisions,person,probablePitcher,stats,homeRuns,previousPlay,game(content(media(epg),summary),tickets),seriesStatus(useOverride=true)&useLatestGames=false&language=en"
            
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                handle_api_failure(response, "Schedule API")
                return []
            
            data = response.json()
            dates = data.get('dates', [])
            if not dates:
                return []
            
            games = dates[0].get('games', [])
            
            # Process each game to ensure probable pitcher data is properly extracted
            processed_games = []
            for game in games:
                # Create a copy of the game data
                processed_game = game.copy()
                
                # Ensure probable pitcher data is properly structured
                teams = processed_game.get('teams', {})
                
                # Process home team probable pitcher
                home_team = teams.get('home', {})
                if 'probablePitcher' in home_team and home_team['probablePitcher']:
                    # Probable pitcher data is already in the correct format from hydrated response
                    pass
                else:
                    # Set empty probable pitcher if not available
                    home_team['probablePitcher'] = {}
                
                # Process away team probable pitcher  
                away_team = teams.get('away', {})
                if 'probablePitcher' in away_team and away_team['probablePitcher']:
                    # Probable pitcher data is already in the correct format from hydrated response
                    pass
                else:
                    # Set empty probable pitcher if not available
                    away_team['probablePitcher'] = {}
                
                processed_games.append(processed_game)
            
            return processed_games
            
        except Exception as e:
            logger.error(f"Error getting schedule for {date}: {e}")
            return []
    
    def get_player_info(self, player_id):
        """Get player information"""
        try:
            url = f"{self.BASE_URL}/people/{player_id}"
            params = {
                "fields": "people,id,firstName,lastName,primaryPosition,code,name,type,abbreviation,batSide,code,description,pitchHand,code,description"
            }
            logger.info(f"[get_player_info] Requesting info for player_id={player_id}")
            response = self.session.get(url, params=params)
            logger.info(f"[get_player_info] Response status: {response.status_code}")
            if response.status_code != 200:
                handle_api_failure(response, "get_player_info")
            data = response.json()
            logger.info(f"[get_player_info] Raw response: {json.dumps(data, indent=2)}")
            people = data.get('people', [])
            if not people or not isinstance(people, list):
                logger.warning(f"[get_player_info] No 'people' field or not a list for player_id={player_id}")
                return None
            player_info = people[0]
            # Validate expected fields
            if not player_info.get('firstName') or not player_info.get('lastName'):
                logger.warning(f"[get_player_info] Missing name fields for player_id={player_id}: {player_info}")
                return None
            logger.info(f"[get_player_info] Extracted player info: {json.dumps(player_info, indent=2)}")
            return player_info
        except Exception as e:
            logger.error(f"Error getting player info for {player_id}: {e}")
            logger.exception("Full traceback:")
            return None
    
    def get_player_stats(self, player_id, season=2024, group='hitting'):
        """Get player stats for the season"""
        try:
            url = f"{self.BASE_URL}/people/{player_id}/stats"
            params = {
                "stats": "season",
                "season": season,
                "group": group,
                "gameType": "R"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                handle_api_failure(response, "get_player_stats")
            
            data = response.json()
            stats_list = data.get('stats', [])
            
            if not stats_list:
                logger.warning(f"No stats found for player {player_id}")
                return pd.DataFrame()
                
            splits = stats_list[0].get('splits', [])
            if not splits:
                logger.warning(f"No splits found for player {player_id}")
                return pd.DataFrame()
                
            stats = splits[0].get('stat', {})
            if not stats:
                logger.warning(f"No stat data found for player {player_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame for easier handling
            stats_df = pd.DataFrame([stats])
            
            # Calculate hit probability for hitters
            if group == 'hitting':
                if 'hits' in stats_df.columns and 'plateAppearances' in stats_df.columns:
                    hit_prob = stats_df['hits'].iloc[0] / stats_df['plateAppearances'].iloc[0]
                else:
                    hit_prob = 0.0
                
                # Log the stats we care about
                logger.info(f"Player {player_id} hitting stats for {season}:")
                logger.info(f"  Hits: {stats.get('hits', 0)}")
                logger.info(f"  Plate Appearances: {stats.get('plateAppearances', 0)}")
                logger.info(f"  Hit Probability: {hit_prob:.3f}")
            else:
                # Log pitching stats
                logger.info(f"Player {player_id} pitching stats for {season}:")
                logger.info(f"  ERA: {stats.get('era', 0)}")
                logger.info(f"  WHIP: {stats.get('whip', 0)}")
                logger.info(f"  Innings Pitched: {stats.get('inningsPitched', 0)}")
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error getting player stats for {player_id}: {e}")
            logger.exception("Full traceback:")
            return pd.DataFrame()
    
    def get_matchup_stats(self, batter_id, pitcher_id):
        """Get batter vs pitcher matchup statistics - now relies only on Retrosheet data via get_historical_matchup"""
        logger.info(f"get_matchup_stats called for {batter_id} vs {pitcher_id} - redirecting to Retrosheet-only lookup")
        
        # This method now simply returns empty data since we only use Retrosheet data
        # The actual matchup logic is handled in get_historical_matchup which uses Retrosheet
        logger.info(f"No MLB Stats API matchup lookup for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return pd.DataFrame()
    
    def _get_matchup_fallback(self, batter_id, pitcher_id):
        """Fallback method - now disabled since we only use Retrosheet data"""
        logger.info(f"MLB Stats API fallback disabled for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return None
    
    def _get_career_matchup_data(self, batter_id, pitcher_id):
        """Career matchup method - now disabled since we only use Retrosheet data"""
        logger.info(f"MLB Stats API career search disabled for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return None
    
    def _get_game_by_game_matchup_data(self, batter_id, pitcher_id):
        """Game by game matchup method - now disabled since we only use Retrosheet data"""
        logger.info(f"MLB Stats API game-by-game search disabled for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return None
    
    def _get_alternate_matchup_data(self, batter_id, pitcher_id):
        """Alternate matchup method - now disabled since we only use Retrosheet data"""
        logger.info(f"MLB Stats API alternate search disabled for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return None
    
    def _try_legacy_lookup_service(self, batter_id, pitcher_id):
        """Legacy lookup method - now disabled since we only use Retrosheet data"""
        logger.info(f"MLB Stats API legacy lookup disabled for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return None
    
    def get_lineups(self, game_pk):
        """Get lineups for a game"""
        try:
            url = f"{self.BASE_URL}/game/{game_pk}/feed/live"
            params = {
                "fields": "liveData,boxscore,teams,home,away,players,id,fullName,position,code"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code == 404:
                logger.warning(f"Game {game_pk} not found (404 error) - likely future/cancelled game")
                return None
            elif response.status_code != 200:
                logger.error(f"API error for game {game_pk}: {response.status_code}")
                return None
                
            data = response.json()
            boxscore = data.get('liveData', {}).get('boxscore', {})
            teams = boxscore.get('teams', {})
            
            home_players = teams.get('home', {}).get('players', {})
            away_players = teams.get('away', {}).get('players', {})
            
            # Filter for position players
            home_lineup = [
                player for player in home_players.values()
                if player.get('position', {}).get('code') not in ['1', 'TWP']
            ]
            away_lineup = [
                player for player in away_players.values()
                if player.get('position', {}).get('code') not in ['1', 'TWP']
            ]
            
            return {
                'home': home_lineup,
                'away': away_lineup
            }
        except Exception as e:
            logger.error(f"Error getting lineups for game {game_pk}: {e}")
            return None
    
    def get_team_roster(self, team_id):
        """Get team roster"""
        try:
            url = f"{self.BASE_URL}/teams/{team_id}/roster"
            params = {
                "fields": "roster,person,id,fullName,primaryPosition,code"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                handle_api_failure(response, "get_team_roster")
            data = response.json()
            return data.get('roster', [])
        except Exception as e:
            logger.error(f"Error getting roster for team {team_id}: {e}")
            return []
    
    def get_venues(self):
        """Get all MLB venues"""
        try:
            url = f"{self.BASE_URL}/venues"
            params = {
                "fields": "venues,id,name,location,city,state"
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                handle_api_failure(response, "get_venues")
            data = response.json()
            return data.get('venues', [])
        except Exception as e:
            logger.error(f"Error getting venues: {e}")
            return []

class MLBHitTracker:
    # Define successful outcomes (hits)
    SUCCESSFUL_OUTCOMES = {
        'single',
        'double',
        'triple',
        'home_run'
    }

    def __init__(self, prediction_date=None):
        """Initialize MLBHitTracker with prediction date (defaults to tomorrow)"""
        self.session = requests.Session()
        self.prediction_date = self.parse_prediction_date(prediction_date)
        self.current_season = self.prediction_date.year
        
        # Initialize MLB API wrapper
        self.mlb = MLBStatsAPI()
        
        # Initialize Retrosheet integration layer for historical data
        self.retrosheet_integration = None
        if RETROSHEET_INTEGRATION_AVAILABLE:
            try:
                self.retrosheet_integration = create_retrosheet_integration_layer(self.current_season)
                logger.info("Retrosheet integration layer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Retrosheet integration: {e}")
                self.retrosheet_integration = None
        else:
            logger.info("Retrosheet integration not available - using MLB Stats API only")
        
        # Initialize ballpark factors
        self.ballpark_factors = self.load_ballpark_factors()
        
        # Default prediction weights - can be optimized later
        self.prediction_weights = {
            'season_avg': 0.4,
            'recent_performance': 0.35,
            'ballpark_factor': 0.15,
            'weather_impact': 0.05,
            'matchup_advantage': 0.05
        }
        
        self.weather_client = None  # Initialize as None, we'll create it when needed
        self.geolocator = Nominatim(user_agent="mlb_hit_tracker")
        
        # Initialize database
        self.init_database()
        
        # Initialize platoon statistics
        self.platoon_stats = self.calculate_platoon_advantages()
        
        # Initialize weather thresholds
        self.weather_thresholds = self.calculate_weather_thresholds()
        
        # Ensure the ballpark factors are loaded
        if not self.ballpark_factors:
            logger.warning("Ballpark factors not loaded properly")
        
        # Initialize training system and optimize weights if sufficient data exists
        self.initialize_training_system()
    
    def init_database(self):
        """Initialize the database connection"""
        try:
            setup_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def load_ballpark_factors(self):
        """Load ballpark factors from database or calculate if not available"""
        try:
            conn = sqlite3.connect('mlb_stats.db')
            c = conn.cursor()
            c.execute('SELECT venue_name, factor FROM park_factors')
            factors = dict(c.fetchall())
            conn.close()
            
            if factors:
                logger.info(f"Loaded {len(factors)} ballpark factors from database")
                return factors
            else:
                logger.info("No ballpark factors in database, calculating...")
                return self.calculate_ballpark_factors()
        except Exception as e:
            logger.error(f"Error loading ballpark factors: {e}")
            logger.info("Calculating ballpark factors...")
            return self.calculate_ballpark_factors()
    
    def parse_prediction_date(self, prediction_date):
        """Parse and validate prediction date"""
        if prediction_date:
            if isinstance(prediction_date, str):
                try:
                    return datetime.strptime(prediction_date, '%Y-%m-%d')
                except ValueError:
                    logger.error("Invalid date format. Please use YYYY-MM-DD format.")
                    raise
            return prediction_date
        else:
            # Default to tomorrow
            return datetime.now() + timedelta(days=1)

    def calculate_venue_factor(self, venue, start_year, end_year):
        """Calculate park factor for a single venue using League Average Method"""
        venue_name = venue.get('name')
        venue_id = venue.get('id')
        if not venue_name or not venue_id:
            return venue_name, 1.0
            
        with timer(f"Venue factor calculation for {venue_name}"):
            # Check Redis cache first
            cache_key = f"park_factor:{venue_id}:{start_year}-{end_year}"
            cached_factor = redis_client.get(cache_key)
            if cached_factor:
                logger.info(f"Redis cache hit for {venue_name}")
                return venue_name, float(cached_factor)
            
            # Check database next
            try:
                conn = sqlite3.connect('mlb_stats.db')
                c = conn.cursor()
                c.execute('''
                    SELECT factor FROM park_factors 
                    WHERE venue_id = ? AND last_updated > ?
                ''', (venue_id, datetime.now() - timedelta(days=1)))
                db_factor = c.fetchone()
                
                if db_factor:
                    conn.close()
                    # Cache in Redis
                    redis_client.setex(cache_key, CACHE_EXPIRATION, str(db_factor[0]))
                    logger.info(f"Database hit for {venue_name}")
                    return venue_name, db_factor[0]
            except Exception as e:
                logger.error(f"Database error for {venue_name}: {e}")
                if 'conn' in locals():
                    conn.close()
                
            # Calculate park factor using League Average Method
            try:
                # First, calculate league average hits per game for the time period
                league_total_hits = 0
                league_total_games = 0
                ballpark_total_hits = 0
                ballpark_total_games = 0
                
                # Get all teams for the calculation period
                teams_url = f"{self.mlb.BASE_URL}/teams"
                teams_params = {
                    "sportId": 1,
                    "season": end_year,
                    "fields": "teams,id,name"
                }
                
                teams_response = requests.get(teams_url, params=teams_params)
                if teams_response.status_code != 200:
                    logger.warning(f"Could not get teams data for venue factor calculation: {teams_response.status_code}")
                    return venue_name, 1.0
                
                teams_data = teams_response.json()
                all_teams = teams_data.get('teams', [])
                
                if not all_teams:
                    logger.warning(f"No teams found for venue factor calculation")
                    return venue_name, 1.0
                
                # Calculate league averages and ballpark-specific stats
                for year in range(start_year, end_year + 1):
                    for team in all_teams:
                        team_id = team.get('id')
                        if not team_id:
                            continue
                            
                        # Get team's overall season hitting stats for league average calculation
                        season_stats_url = f"{self.mlb.BASE_URL}/teams/{team_id}/stats"
                        season_params = {
                            "stats": "season",
                            "season": year,
                            "group": "hitting",
                            "gameType": "R"
                        }
                        
                        season_response = requests.get(season_stats_url, params=season_params)
                        if season_response.status_code == 200:
                            season_data = season_response.json()
                            for stat in season_data.get('stats', []):
                                for split in stat.get('splits', []):
                                    stat_data = split.get('stat', {})
                                    hits = stat_data.get('hits', 0)
                                    games = stat_data.get('gamesPlayed', 0)
                                    if games > 0:  # Only count if they actually played games
                                        league_total_hits += hits
                                        league_total_games += games
                        
                        # Get team hitting stats when playing at this specific ballpark
                        ballpark_stats_url = f"{self.mlb.BASE_URL}/teams/{team_id}/stats"
                        ballpark_params = {
                            "stats": "statSplits",
                            "season": year,
                            "group": "hitting",
                            "gameType": "R",
                            "sitCodes": f"v{venue_id}"  # Games at this specific venue
                        }
                        
                        ballpark_response = requests.get(ballpark_stats_url, params=ballpark_params)
                        if ballpark_response.status_code == 200:
                            ballpark_data = ballpark_response.json()
                            for stat in ballpark_data.get('stats', []):
                                for split in stat.get('splits', []):
                                    stat_data = split.get('stat', {})
                                    hits = stat_data.get('hits', 0)
                                    games = stat_data.get('gamesPlayed', 0)
                                    if games > 0:  # Only count if they actually played games
                                        ballpark_total_hits += hits
                                        ballpark_total_games += games
                
                # Calculate park factor using League Average Method
                if league_total_games > 0 and ballpark_total_games > 0:
                    league_hits_per_game = league_total_hits / league_total_games
                    ballpark_hits_per_game = ballpark_total_hits / ballpark_total_games
                    
                    if league_hits_per_game > 0:
                        park_factor = ballpark_hits_per_game / league_hits_per_game
                        park_factor = round(park_factor, 3)
                    else:
                        park_factor = 1.0
                    
                    logger.info(f"Calculated park factor for {venue_name}: {park_factor}")
                    logger.info(f"  Ballpark: {ballpark_hits_per_game:.3f} hits/game ({ballpark_total_games} games)")
                    logger.info(f"  League avg: {league_hits_per_game:.3f} hits/game ({league_total_games} total games)")
                else:
                    logger.warning(f"Insufficient data for {venue_name} (ballpark: {ballpark_total_games} games, league: {league_total_games} games)")
                    park_factor = 1.0
                    
                # Store in database and cache
                try:
                    conn = sqlite3.connect('mlb_stats.db')
                    c = conn.cursor()
                    c.execute('''
                        INSERT OR REPLACE INTO park_factors (venue_id, venue_name, factor, last_updated)
                        VALUES (?, ?, ?, ?)
                    ''', (venue_id, venue_name, park_factor, datetime.now()))
                    conn.commit()
                    conn.close()
                    
                    # Cache in Redis
                    redis_client.setex(cache_key, CACHE_EXPIRATION, str(park_factor))
                except Exception as e:
                    logger.error(f"Error storing park factor for {venue_name}: {e}")
                    if 'conn' in locals():
                        conn.close()
                
                return venue_name, park_factor
                
            except Exception as e:
                logger.error(f"Error calculating park factor for {venue_name}: {e}")
                return venue_name, 1.0

    def calculate_ballpark_factors(self, years_back=3):
        """Calculate park factors for all venues with enhanced Retrosheet integration"""
        with timer("Ballpark factors calculation"):
            try:
                # Get current year
                current_year = datetime.now().year
                start_year = current_year - years_back
                end_year = current_year - 1
                
                # Try Retrosheet data first for historical ballpark factors
                if self.retrosheet_integration:
                    logger.info("Attempting to use Retrosheet data for ballpark factors")
                    try:
                        # Get active ballparks list
                        active_ballparks = {
                            'American Family Field',
                            'Angel Stadium',
                            'Busch Stadium',
                            'Chase Field',
                            'Citi Field',
                            'Citizens Bank Park',
                            'Comerica Park',
                            'Coors Field',
                            'Dodger Stadium',
                            'Fenway Park',
                            'Globe Life Field',
                            'Great American Ball Park',
                            'Guaranteed Rate Field',
                            'Kauffman Stadium',
                            'LoanDepot Park',
                            'Minute Maid Park',
                            'Nationals Park',
                            'Oakland Coliseum',
                            'Oracle Park',
                            'Oriole Park at Camden Yards',
                            'Petco Park',
                            'PNC Park',
                            'Progressive Field',
                            'Rogers Centre',
                            'T-Mobile Park',
                            'Target Field',
                            'Tropicana Field',
                            'Truist Park',
                            'Wrigley Field',
                            'Yankee Stadium',
                            'Daikin Park'  # Houston's new stadium name
                        }
                        
                        venue_factors = {}
                        retrosheet_success_count = 0
                        
                        for ballpark_name in tqdm(active_ballparks, desc="Processing ballparks with Retrosheet"):
                            try:
                                retrosheet_factors = self.retrosheet_integration.get_historical_ballpark_factors(
                                    ballpark_name, years_back
                                )
                                
                                if (retrosheet_factors and 
                                    retrosheet_factors.get('source') == 'retrosheet_optimized' and
                                    retrosheet_factors.get('sample_size', 0) > 0):
                                    
                                    venue_factors[ballpark_name] = retrosheet_factors['park_factor']
                                    retrosheet_success_count += 1
                                    logger.info(f"Retrosheet park factor for {ballpark_name}: {retrosheet_factors['park_factor']:.3f} "
                                              f"(confidence: {retrosheet_factors['confidence']:.2f}, "
                                              f"sample_size: {retrosheet_factors['sample_size']})")
                                else:
                                    logger.debug(f"No Retrosheet park factor data for {ballpark_name}")
                                    
                            except Exception as e:
                                logger.warning(f"Error getting Retrosheet park factor for {ballpark_name}: {e}")
                        
                        if retrosheet_success_count > 0:
                            logger.info(f"Successfully calculated {retrosheet_success_count} ballpark factors using Retrosheet data")
                            
                            # Store Retrosheet factors in database
                            try:
                                conn = sqlite3.connect('mlb_stats.db')
                                c = conn.cursor()
                                for venue_name, factor in venue_factors.items():
                                    c.execute('''
                                        INSERT OR REPLACE INTO park_factors (venue_name, factor, last_updated)
                                        VALUES (?, ?, ?)
                                    ''', (venue_name, factor, datetime.now()))
                                conn.commit()
                                conn.close()
                                logger.info(f"Stored {len(venue_factors)} Retrosheet ballpark factors in database")
                            except Exception as e:
                                logger.error(f"Error storing Retrosheet ballpark factors: {e}")
                                if 'conn' in locals():
                                    conn.close()
                            
                            # If we got most ballparks from Retrosheet, return those
                            if retrosheet_success_count >= len(active_ballparks) * 0.7:  # 70% success rate
                                logger.info(f"Using Retrosheet ballpark factors ({retrosheet_success_count}/{len(active_ballparks)} ballparks)")
                                return venue_factors
                            else:
                                logger.info(f"Retrosheet coverage insufficient ({retrosheet_success_count}/{len(active_ballparks)}), falling back to MLB API")
                        else:
                            logger.info("No Retrosheet ballpark factors available, falling back to MLB API")
                            
                    except Exception as e:
                        logger.warning(f"Error using Retrosheet for ballpark factors: {e}")
                
                # Fallback to existing MLB Stats API method
                logger.info("Using MLB Stats API for ballpark factors calculation")
                
                # Get all venues
                venues = self.mlb.get_venues()
                if not venues:
                    logger.error("No venues found")
                    return {}
                
                # Filter for active MLB venues only
                mlb_venues = []
                active_ballparks = {
                    'American Family Field',
                    'Angel Stadium',
                    'Busch Stadium',
                    'Chase Field',
                    'Citi Field',
                    'Citizens Bank Park',
                    'Comerica Park',
                    'Coors Field',
                    'Dodger Stadium',
                    'Fenway Park',
                    'Globe Life Field',
                    'Great American Ball Park',
                    'Guaranteed Rate Field',
                    'Kauffman Stadium',
                    'LoanDepot Park',
                    'Minute Maid Park',
                    'Nationals Park',
                    'Oakland Coliseum',
                    'Oracle Park',
                    'Oriole Park at Camden Yards',
                    'Petco Park',
                    'PNC Park',
                    'Progressive Field',
                    'Rogers Centre',
                    'T-Mobile Park',
                    'Target Field',
                    'Tropicana Field',
                    'Truist Park',
                    'Wrigley Field',
                    'Yankee Stadium',
                    'Daikin Park'  # Houston's new stadium name
                }
                
                for venue in venues:
                    venue_name = venue.get('name', '')
                    if venue_name in active_ballparks:
                        mlb_venues.append(venue)
                        
                logger.info(f"Processing {len(mlb_venues)} active MLB venues out of {len(venues)} total venues")
                    
                # Calculate factors for each venue
                venue_factors = {}
                for venue in tqdm(mlb_venues, desc="Processing MLB venues"):
                    try:
                        venue_name, factor = self.calculate_venue_factor(venue, start_year, end_year)
                        if venue_name and factor != 1.0:
                            venue_factors[venue_name] = factor
                    except Exception as e:
                        logger.error(f"Error processing venue {venue.get('name', 'unknown')}: {e}")
                        continue
                
                # Store in database
                try:
                    conn = sqlite3.connect('mlb_stats.db')
                    c = conn.cursor()
                    for venue_name, factor in venue_factors.items():
                        c.execute('''
                            INSERT OR REPLACE INTO park_factors (venue_name, factor, last_updated)
                            VALUES (?, ?, ?)
                        ''', (venue_name, factor, datetime.now()))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error storing ballpark factors: {e}")
                    if 'conn' in locals():
                        conn.close()
                
                return venue_factors
                
            except Exception as e:
                logger.error(f"Error calculating ballpark factors: {e}")
                return {}

    def calculate_platoon_advantages(self, years_back=3):
        """Calculate platoon advantages based on historical data with improved error handling"""
        try:
            end_year = self.current_season - 1
            start_year = end_year - years_back + 1
            
            # Check Redis cache first
            cache_key = f"platoon_advantages:{start_year}:{end_year}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                cached_result = json.loads(cached_data)
                # Check if cached data is valid (not all zeros)
                if any(value > 0 for value in cached_result.values()):
                    logger.info(f"Redis cache hit for platoon advantages: {start_year}-{end_year}")
                    return cached_result
                else:
                    logger.warning("Cached platoon data appears corrupted (all zeros), recalculating...")
            
            logger.info(f"Calculating platoon advantages from Statcast data ({start_year}-{end_year})")
            
            # Initialize counters
            lhb_vs_rhp = {'hits': 0, 'plate_appearances': 0}  # Left-handed batter vs right-handed pitcher
            lhb_vs_lhp = {'hits': 0, 'plate_appearances': 0}  # Left-handed batter vs left-handed pitcher
            rhb_vs_rhp = {'hits': 0, 'plate_appearances': 0}  # Right-handed batter vs right-handed pitcher
            rhb_vs_lhp = {'hits': 0, 'plate_appearances': 0}  # Right-handed batter vs left-handed pitcher
            
            successful_years = 0
            years = list(range(start_year, end_year + 1))
            
            with tqdm(years, desc="Processing platoon stats by year") as pbar:
                for year in pbar:
                    try:
                        # Check Redis cache for year's data
                        year_cache_key = f"statcast_year:{year}"
                        cached_year_data = redis_client.get(year_cache_key)
                        
                        if cached_year_data:
                            logger.info(f"Redis cache hit for {year} Statcast data")
                            data = pd.read_json(StringIO(cached_year_data))
                        else:
                            # Get season data
                            pbar.set_description(f"Fetching {year} season data")
                            logger.info(f"Fetching Statcast data for {year}")
                            data = statcast(start_dt=f"{year}-03-01", end_dt=f"{year}-11-01")
                            if data is not None and not data.empty:
                                # Reset index to avoid duplicate index issues
                                data = data.reset_index(drop=True)
                                # Cache the year's data
                                redis_client.setex(year_cache_key, CACHE_EXPIRATION, data.to_json())
                                logger.info(f"Successfully cached {len(data)} records for {year}")
                            
                        if data is None or data.empty:
                            logger.warning(f"No Statcast data available for {year}")
                            continue
                        
                        # Ensure index is unique
                        if not data.index.is_unique:
                            data = data.reset_index(drop=True)
                            logger.info(f"Reset duplicate indices for {year} data")
                        
                        year_data_processed = 0
                        
                        # Filter for completed plate appearances with better criteria
                        completed_pas = data[
                            (data['events'].notna()) &  # Must have an event
                            (data['events'] != '') &    # Event must not be empty
                            (data['stand'].notna()) &   # Must have batter handedness
                            (data['p_throws'].notna())  # Must have pitcher handedness
                        ]
                        
                        logger.info(f"Processing {len(completed_pas)} completed PAs for {year}")
                        
                        # Group by batter and pitcher handedness
                        for _, row in completed_pas.iterrows():
                            try:
                                # Determine if it's a hit
                                event = row.get('events')
                                is_hit = event in self.SUCCESSFUL_OUTCOMES
                                
                                batter_side = row['stand']
                                pitcher_side = row['p_throws']
                                
                                # Update appropriate counter based on handedness
                                if batter_side == 'L':  # Left-handed batter
                                    if pitcher_side == 'R':  # vs right-handed pitcher
                                        lhb_vs_rhp['plate_appearances'] += 1
                                        if is_hit:
                                            lhb_vs_rhp['hits'] += 1
                                    else:  # vs left-handed pitcher
                                        lhb_vs_lhp['plate_appearances'] += 1
                                        if is_hit:
                                            lhb_vs_lhp['hits'] += 1
                                else:  # Right-handed batter
                                    if pitcher_side == 'R':  # vs right-handed pitcher
                                        rhb_vs_rhp['plate_appearances'] += 1
                                        if is_hit:
                                            rhb_vs_rhp['hits'] += 1
                                    else:  # vs left-handed pitcher
                                        rhb_vs_lhp['plate_appearances'] += 1
                                        if is_hit:
                                            rhb_vs_lhp['hits'] += 1
                                
                                year_data_processed += 1
                                
                            except Exception as e:
                                logger.debug(f"Error processing row for {year}: {e}")
                                continue
                        
                        if year_data_processed > 0:
                            successful_years += 1
                            logger.info(f"Successfully processed {year_data_processed} PAs for {year}")
                        
                    except Exception as e:
                        logger.error(f"Error fetching/processing statcast data for {year}: {e}")
                        continue
            
            # Calculate total PAs processed
            total_pas = (lhb_vs_rhp['plate_appearances'] + lhb_vs_lhp['plate_appearances'] + 
                        rhb_vs_rhp['plate_appearances'] + rhb_vs_lhp['plate_appearances'])
            
            logger.info(f"Platoon calculation summary:")
            logger.info(f"  Successful years: {successful_years}/{len(years)}")
            logger.info(f"  Total PAs processed: {total_pas:,}")
            logger.info(f"  L vs R: {lhb_vs_rhp['plate_appearances']:,} PAs")
            logger.info(f"  L vs L: {lhb_vs_lhp['plate_appearances']:,} PAs")
            logger.info(f"  R vs R: {rhb_vs_rhp['plate_appearances']:,} PAs")
            logger.info(f"  R vs L: {rhb_vs_lhp['plate_appearances']:,} PAs")
            
            # Calculate averages with minimum threshold check
            min_pas_threshold = 1000  # Minimum PAs needed for reliable average
            
            # Calculate averages
            with tqdm(total=4, desc="Calculating platoon averages") as pbar:
                lhb_vs_rhp_avg = (lhb_vs_rhp['hits'] / lhb_vs_rhp['plate_appearances'] 
                                 if lhb_vs_rhp['plate_appearances'] >= min_pas_threshold else 0.260)
                pbar.update(1)
                
                lhb_vs_lhp_avg = (lhb_vs_lhp['hits'] / lhb_vs_lhp['plate_appearances'] 
                                 if lhb_vs_lhp['plate_appearances'] >= min_pas_threshold else 0.240)
                pbar.update(1)
                
                rhb_vs_rhp_avg = (rhb_vs_rhp['hits'] / rhb_vs_rhp['plate_appearances'] 
                                 if rhb_vs_rhp['plate_appearances'] >= min_pas_threshold else 0.250)
                pbar.update(1)
                
                rhb_vs_lhp_avg = (rhb_vs_lhp['hits'] / rhb_vs_lhp['plate_appearances'] 
                                 if rhb_vs_lhp['plate_appearances'] >= min_pas_threshold else 0.270)
                pbar.update(1)
            
            result = {
                'L_vs_R': round(lhb_vs_rhp_avg, 3),
                'L_vs_L': round(lhb_vs_lhp_avg, 3),
                'R_vs_R': round(rhb_vs_rhp_avg, 3),
                'R_vs_L': round(rhb_vs_lhp_avg, 3)
            }
            
            # Only cache if we have reasonable data
            if total_pas > 10000 or successful_years >= 1:
                redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
                logger.info(f"Cached platoon advantages: {result}")
            else:
                logger.warning(f"Insufficient data for reliable platoon stats, using defaults")
                result = self._get_default_platoon_stats()
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating platoon advantages: {e}")
            logger.info("Using default platoon statistics")
            return self._get_default_platoon_stats()
    
    def _get_default_platoon_stats(self):
        """Get default platoon statistics based on historical MLB averages"""
        return {
            'L_vs_R': 0.260,  # Left-handed batters typically do well vs righties
            'L_vs_L': 0.240,  # Struggle more vs lefties
            'R_vs_R': 0.250,  # Right-handed batters do okay vs righties
            'R_vs_L': 0.270   # Do better vs lefties (platoon advantage)
        }

    def calculate_weather_thresholds(self, years_back=3):
        """Calculate optimal weather thresholds based on actual hitting performance"""
        try:
            end_year = self.current_season - 1
            start_year = end_year - years_back + 1
            
            # Check Redis cache first
            cache_key = f"weather_thresholds:{start_year}:{end_year}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for weather thresholds: {start_year}-{end_year}")
                return json.loads(cached_data)
            
            logger.info(f"Calculating weather thresholds based on hitting performance ({start_year}-{end_year})")
            
            weather_hitting_data = []
            
            # Collect historical weather and hitting data
            for year in range(start_year, end_year + 1):
                logger.info(f"Processing {year} data for weather threshold calculation...")
                
                # Get statcast data for the year to correlate weather with hitting
                try:
                    # Sample key dates throughout the season to get varied weather conditions
                    sample_dates = [
                        f"{year}-04-15", f"{year}-05-15", f"{year}-06-15",
                        f"{year}-07-15", f"{year}-08-15", f"{year}-09-15"
                    ]
                    
                    for date_str in sample_dates:
                        # Get games for this date
                        games = self.mlb.get_schedule_by_date(date_str)
                        
                        for game in games[:5]:  # Limit to 5 games per date to manage processing
                            try:
                                # Get box score to see if game was completed
                                box_score = self.get_game_box_score(game.get('gamePk'))
                                if not box_score:
                                    continue
                                
                                # Extract weather data if available
                                venue = game.get('venue', {})
                                venue_name = venue.get('name')
                                if not venue_name:
                                    continue
                                
                                # Get weather location for this venue
                                weather_location = self.get_weather_location_for_venue(
                                    venue.get('id'), venue_name
                                )
                                if not weather_location:
                                    continue
                                
                                # Get coordinates for weather lookup (parse or use venue defaults)
                                lat, lon = 40.0, -74.0  # Default to NYC area
                                if ',' in weather_location:
                                    parts = weather_location.split(',')
                                    if len(parts) >= 2:
                                        try:
                                            lat = float(parts[0].strip())
                                            lon = float(parts[1].strip())
                                        except:
                                            # Use venue-based coordinates if available
                                            venue_coords = {
                                                'Yankee Stadium': (40.8296, -73.9262),
                                                'Fenway Park': (42.3467, -71.0972),
                                                'Wrigley Field': (41.9484, -87.6553),
                                                'Dodger Stadium': (34.0739, -118.2400),
                                                'Oracle Park': (37.7786, -122.3893),
                                                'Coors Field': (39.7559, -104.9942)
                                            }
                                            coords = venue_coords.get(venue_name, (40.0, -74.0))
                                            lat, lon = coords
                                
                                # Convert date string to datetime object
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                
                                # Get realistic historical weather data
                                estimated_weather = self._estimate_historical_weather(lat, lon, date_obj)
                                if not estimated_weather:
                                    continue
                                
                                # Calculate hitting performance for this game
                                game_hitting_stats = self._extract_game_hitting_performance(box_score)
                                if not game_hitting_stats:
                                    continue
                                
                                # Combine weather and hitting data
                                weather_hitting_data.append({
                                    'temperature': estimated_weather['temperature'],
                                    'wind_speed': estimated_weather['wind_speed'],
                                    'conditions': 'Rain' if estimated_weather.get('precipitation', False) else 'Clear',
                                    'humidity': estimated_weather.get('humidity', 50),
                                    'pressure': estimated_weather.get('pressure', 29.92),
                                    'hits_per_ab': game_hitting_stats['hits_per_ab'],
                                    'total_hits': game_hitting_stats['total_hits'],
                                    'total_ab': game_hitting_stats['total_ab'],
                                    'date': date_str,
                                    'venue': venue_name,
                                    'weather_source': estimated_weather.get('source', 'unknown')
                                })
                                
                            except Exception as e:
                                logger.debug(f"Error processing game {game.get('gamePk')}: {e}")
                                continue
                                
                except Exception as e:
                    logger.warning(f"Error processing {year} weather threshold data: {e}")
                    continue
            
            if len(weather_hitting_data) < 20:
                logger.warning(f"Insufficient weather-hitting data ({len(weather_hitting_data)} records), using enhanced defaults")
                return self._get_enhanced_default_thresholds()
            
            # Analyze the data to find optimal thresholds
            df = pd.DataFrame(weather_hitting_data)
            logger.info(f"Analyzing {len(df)} weather-hitting records for threshold optimization")
            
            # Find temperature thresholds based on hitting performance
            temps = np.sort(df['temperature'].unique())
            best_high_temp = 75  # default
            best_low_temp = 55   # default
            max_hitting_variance = 0
            
            for temp in temps:
                if len(temps) < 10:  # Need sufficient data points
                    continue
                    
                cold_games = df[df['temperature'] < temp]
                warm_games = df[df['temperature'] >= temp]
                
                if len(cold_games) >= 5 and len(warm_games) >= 5:  # Minimum sample size
                    cold_avg = cold_games['hits_per_ab'].mean()
                    warm_avg = warm_games['hits_per_ab'].mean()
                    hitting_variance = abs(warm_avg - cold_avg)
                    
                    if hitting_variance > max_hitting_variance:
                        max_hitting_variance = hitting_variance
                        if temp > 70:
                            best_high_temp = temp
                        elif temp < 65:
                            best_low_temp = temp
            
            # Find wind speed threshold based on hitting performance
            winds = np.sort(df['wind_speed'].unique())
            best_wind = 10  # default
            max_wind_hitting_effect = 0
            
            for wind in winds:
                if len(winds) < 8:  # Need sufficient data points
                    continue
                    
                calm_games = df[df['wind_speed'] < wind]
                windy_games = df[df['wind_speed'] >= wind]
                
                if len(calm_games) >= 5 and len(windy_games) >= 5:  # Minimum sample size
                    calm_avg = calm_games['hits_per_ab'].mean()
                    windy_avg = windy_games['hits_per_ab'].mean()
                    wind_effect = abs(calm_avg - windy_avg)
                    
                    if wind_effect > max_wind_hitting_effect:
                        max_wind_hitting_effect = wind_effect
                        best_wind = wind
            
            # Calculate precipitation impact
            if 'rain' in df['conditions'].str.lower().values:
                rain_games = df[df['conditions'].str.lower().str.contains('rain', na=False)]
                clear_games = df[~df['conditions'].str.lower().str.contains('rain', na=False)]
                
                if len(rain_games) >= 3 and len(clear_games) >= 10:
                    rain_avg = rain_games['hits_per_ab'].mean()
                    clear_avg = clear_games['hits_per_ab'].mean()
                    precipitation_impact = rain_avg / clear_avg if clear_avg > 0 else 1.0
                else:
                    precipitation_impact = 0.97  # Default
            else:
                precipitation_impact = 0.97  # Default
            
            result = {
                'high_temp': int(best_high_temp),
                'low_temp': int(best_low_temp),
                'wind_speed': int(best_wind),
                'precipitation_factor': round(precipitation_impact, 3),
                'data_points': len(weather_hitting_data),
                'hitting_variance_found': round(max_hitting_variance, 4),
                'wind_effect_found': round(max_wind_hitting_effect, 4)
            }
            
            logger.info(f"Calculated weather thresholds from hitting performance:")
            logger.info(f"  High temp: {result['high_temp']}°F")
            logger.info(f"  Low temp: {result['low_temp']}°F") 
            logger.info(f"  Wind speed: {result['wind_speed']}mph")
            logger.info(f"  Precipitation factor: {result['precipitation_factor']}")
            logger.info(f"  Based on {result['data_points']} game records")
            
            # Cache for 24 hours
            redis_client.setex(cache_key, 60 * 60 * 24, json.dumps(result))
            return result
            
        except Exception as e:
            logger.error(f"Error calculating weather thresholds: {e}")
            logger.exception("Full traceback:")
            return self._get_enhanced_default_thresholds()

    def _estimate_historical_weather(self, lat, lon, date):
        """Get historical weather averages based on location and time of year"""
        try:
            month = date.month
            day_of_year = date.timetuple().tm_yday
            
            # Climate-based temperature averages by latitude and season
            # Using NOAA climate normals as reference
            if lat >= 45:  # Northern regions (e.g., Minneapolis, Seattle)
                if 3 <= month <= 5:  # Spring
                    base_temp = 50
                elif 6 <= month <= 8:  # Summer
                    base_temp = 75
                elif 9 <= month <= 11:  # Fall
                    base_temp = 55
                else:  # Winter
                    base_temp = 25
            elif lat >= 35:  # Mid-latitude regions (e.g., Chicago, New York)
                if 3 <= month <= 5:  # Spring
                    base_temp = 60
                elif 6 <= month <= 8:  # Summer
                    base_temp = 80
                elif 9 <= month <= 11:  # Fall
                    base_temp = 65
                else:  # Winter
                    base_temp = 40
            elif lat >= 25:  # Southern regions (e.g., Atlanta, Phoenix)
                if 3 <= month <= 5:  # Spring
                    base_temp = 70
                elif 6 <= month <= 8:  # Summer
                    base_temp = 85
                elif 9 <= month <= 11:  # Fall
                    base_temp = 75
                else:  # Winter
                    base_temp = 55
            else:  # Subtropical/tropical (e.g., Miami, Tampa)
                if 3 <= month <= 5:  # Spring
                    base_temp = 75
                elif 6 <= month <= 8:  # Summer
                    base_temp = 85
                elif 9 <= month <= 11:  # Fall
                    base_temp = 80
                else:  # Winter
                    base_temp = 70
            
            # Longitude adjustments for continental effects
            if lon < -100:  # Western regions (drier, more temperature variation)
                if month in [6, 7, 8]:  # Summer - hotter in west
                    base_temp += 5
                elif month in [12, 1, 2]:  # Winter - colder in continental west
                    base_temp -= 5
            elif lon > -80:  # Eastern coastal regions (maritime influence)
                if month in [6, 7, 8]:  # Summer - cooler near coast
                    base_temp -= 3
                elif month in [12, 1, 2]:  # Winter - warmer near coast
                    base_temp += 3
            
            # Climate-based humidity averages
            if lat >= 40:  # Northern regions
                humidity = 65 if month in [4, 5, 6, 10, 11] else 60
            elif lon > -90:  # Eastern regions (higher humidity)
                humidity = 75 if month in [6, 7, 8] else 65
            else:  # Western regions (lower humidity)
                humidity = 45 if month in [6, 7, 8] else 50
            
            # Standard atmospheric pressure (varies slightly with altitude)
            # Estimate altitude effect: -0.01 inches per 300 feet elevation
            estimated_elevation = max(0, (lat - 25) * 100)  # Rough elevation estimate
            pressure = 29.92 - (estimated_elevation / 300 * 0.01)
            
            # Seasonal wind speed averages
            if month in [3, 4, 11, 12]:  # Windier months
                wind_speed = 12
            elif month in [6, 7, 8]:  # Calmer summer months
                wind_speed = 8
            else:
                wind_speed = 10
            
            # Seasonal precipitation likelihood (not random, based on climate)
            if lat >= 40:  # Northern regions
                precip_likely = month in [4, 5, 10, 11]  # Spring/fall precipitation
            else:  # Southern regions
                precip_likely = month in [6, 7, 8, 9] if lon > -100 else month in [12, 1, 2]  # Summer thunderstorms (east) or winter rains (west)
            
            logger.info(f"Climate-based weather average for {date.strftime('%Y-%m-%d')} at ({lat:.2f}, {lon:.2f}): {base_temp}°F, {humidity}% humidity")
            
            return {
                'temperature': base_temp,
                'humidity': humidity,
                'pressure': round(pressure, 2),
                'wind_speed': wind_speed,
                'precipitation': precip_likely,
                'source': 'climate_normal_average'
            }
            
        except Exception as e:
            logger.warning(f"Weather average calculation failed: {e}")
            # Fallback to simple defaults based on time of year
            month = date.month
            temp = 70 if 4 <= month <= 10 else 50  # Warm season vs cool season
            return {
                'temperature': temp,
                'humidity': 60,
                'pressure': 29.92,
                'wind_speed': 10,
                'precipitation': False,
                'source': 'simple_seasonal_default'
            }

    def _extract_game_hitting_performance(self, box_score):
        """Extract hitting performance metrics from a completed game"""
        try:
            total_hits = 0
            total_ab = 0
            
            # Process both teams
            for team_key in ['home', 'away']:
                team_data = box_score.get('teams', {}).get(team_key, {})
                players = team_data.get('players', {})
                
                for player_key, player_data in players.items():
                    if not player_key.startswith('ID'):
                        continue
                    
                    batting_stats = player_data.get('stats', {}).get('batting', {})
                    if batting_stats:
                        hits = batting_stats.get('hits', 0)
                        at_bats = batting_stats.get('atBats', 0)
                        
                        total_hits += hits
                        total_ab += at_bats
            
            if total_ab == 0:
                return None
                
            return {
                'hits_per_ab': total_hits / total_ab,
                'total_hits': total_hits,
                'total_ab': total_ab
            }
            
        except Exception as e:
            logger.debug(f"Error extracting hitting performance: {e}")
            return None

    def _get_enhanced_default_thresholds(self):
        """Get enhanced default weather thresholds based on baseball research"""
        return {
            'high_temp': 78,    # Based on research showing hitting improves above ~78°F
            'low_temp': 50,     # Based on research showing hitting declines below ~50°F
            'wind_speed': 12,   # Research shows 12mph+ winds start affecting ball flight
            'precipitation_factor': 0.96,  # Studies show ~4% hitting decline in rain
            'data_points': 0,
            'hitting_variance_found': 0,
            'wind_effect_found': 0
        }

    def optimize_prediction_weights(self, training_data):
        """Optimize the weights for different prediction factors using enhanced training data"""
        try:
            if not training_data or len(training_data) < 10:
                logger.warning("Insufficient training data for optimization, using default weights")
                return {
                    'season_avg': 0.4,
                    'recent_performance': 0.35,
                    'ballpark_factor': 0.15,
                    'weather_impact': 0.05,
                    'matchup_advantage': 0.05
                }
                
            # Convert training data to DataFrame
            df = pd.DataFrame(training_data)
            logger.info(f"Optimizing weights with {len(df)} training records")
            
            # Log some basic statistics about the training data
            if 'predicted_prob' in df.columns:
                logger.info(f"Predicted probability range: {df['predicted_prob'].min():.3f} - {df['predicted_prob'].max():.3f}")
                logger.info(f"Mean predicted probability: {df['predicted_prob'].mean():.3f}")
            
            if 'got_hit' in df.columns:
                hit_rate = df['got_hit'].mean()
                logger.info(f"Actual hit rate in training data: {hit_rate:.3f}")
            
            # Calculate correlation coefficients for available factors
            correlations = {}
            
            # Season average correlation
            if 'season_avg' in df.columns and 'got_hit' in df.columns:
                season_corr = df['season_avg'].corr(df['got_hit'])
                if not pd.isna(season_corr):
                    correlations['season_avg'] = abs(season_corr)
                    logger.info(f"Season average correlation with hits: {season_corr:.3f}")
                    
            # Recent performance correlation
            if 'recent_avg' in df.columns and 'got_hit' in df.columns:
                recent_corr = df['recent_avg'].corr(df['got_hit'])
                if not pd.isna(recent_corr):
                    correlations['recent_avg'] = abs(recent_corr)
                    logger.info(f"Recent average correlation with hits: {recent_corr:.3f}")
                    
            # Overall prediction correlation
            if 'predicted_prob' in df.columns and 'got_hit' in df.columns:
                pred_corr = df['predicted_prob'].corr(df['got_hit'])
                if not pd.isna(pred_corr):
                    correlations['predicted_prob'] = abs(pred_corr)
                    logger.info(f"Predicted probability correlation with hits: {pred_corr:.3f}")
            
            # Ballpark factor correlation (if we have variation)
            if 'ballpark_factor' in df.columns and 'got_hit' in df.columns:
                ballpark_var = df['ballpark_factor'].var()
                if ballpark_var > 0.001:  # Only if there's meaningful variation
                    ballpark_corr = df['ballpark_factor'].corr(df['got_hit'])
                    if not pd.isna(ballpark_corr):
                        correlations['ballpark_factor'] = abs(ballpark_corr)
                        logger.info(f"Ballpark factor correlation with hits: {ballpark_corr:.3f}")
            
            # Check if we have meaningful correlations to work with
            valid_correlations = {k: v for k, v in correlations.items() if v > 0.001}
            
            if len(valid_correlations) < 2:
                logger.warning("Insufficient correlation data for optimization, using default weights")
                logger.info(f"Available correlations: {correlations}")
                return {
                    'season_avg': 0.4,
                    'recent_performance': 0.35,
                    'ballpark_factor': 0.15,
                    'weather_impact': 0.05,
                    'matchup_advantage': 0.05
                }
            
            # Calculate prediction accuracy metrics
            if 'predicted_prob' in df.columns and 'got_hit' in df.columns:
                # Calculate mean absolute error
                mae = df['prediction_error'].mean() if 'prediction_error' in df.columns else None
                if mae is not None:
                    logger.info(f"Mean Absolute Error: {mae:.3f}")
                
                # Calculate how often we correctly predicted hits vs no-hits
                correct_predictions = ((df['predicted_prob'] > 0.5) == (df['got_hit'] == 1)).sum()
                accuracy = correct_predictions / len(df)
                logger.info(f"Binary prediction accuracy: {accuracy:.3f}")
            
            # Optimize weights based on correlations
            # Focus on the factors that show the strongest correlation with actual outcomes
            total_corr = sum(valid_correlations.values())
            
            if total_corr > 0:
                # Calculate optimized weights for main factors (85% of total weight)
                main_weight_total = 0.85
                
                # Map correlation keys to weight keys
                season_weight = valid_correlations.get('season_avg', 0.1) / total_corr * main_weight_total
                recent_weight = valid_correlations.get('recent_avg', 0.1) / total_corr * main_weight_total
                
                # Handle predicted_prob correlation (this is overall prediction quality)
                # If overall prediction quality is good, maintain current balance
                # If poor, adjust toward the best individual factors
                pred_quality = valid_correlations.get('predicted_prob', 0)
                
                if pred_quality > 0.1:  # Good overall prediction quality
                    # Fine-tune existing weights
                    adjustment_factor = min(1.2, 1 + pred_quality)
                    season_weight *= adjustment_factor if 'season_avg' in valid_correlations else 1
                    recent_weight *= adjustment_factor if 'recent_avg' in valid_correlations else 1
                
                # Normalize to ensure weights sum to main_weight_total
                total_main_weight = season_weight + recent_weight
                if total_main_weight > 0:
                    season_weight = (season_weight / total_main_weight) * main_weight_total
                    recent_weight = (recent_weight / total_main_weight) * main_weight_total
                else:
                    # Fallback to default proportions
                    season_weight = 0.4 * main_weight_total
                    recent_weight = 0.35 * main_weight_total
                
                # Adjust ballpark factor weight based on its correlation
                ballpark_weight = 0.10  # Default
                if 'ballpark_factor' in valid_correlations:
                    ballpark_correlation = valid_correlations['ballpark_factor']
                    if ballpark_correlation > 0.05:
                        ballpark_weight = min(0.20, 0.10 + ballpark_correlation * 0.5)
                
                weights = {
                    'season_avg': round(season_weight, 3),
                    'recent_performance': round(recent_weight, 3),
                    'ballpark_factor': round(ballpark_weight, 3),
                    'weather_impact': 0.05,   # Keep weather impact stable
                    'matchup_advantage': round(max(0.02, 1 - season_weight - recent_weight - ballpark_weight - 0.05), 3)
                }
                
                # Ensure weights sum to 1.0
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    # Normalize weights
                    for key in weights:
                        weights[key] = round(weights[key] / total_weight, 3)
                
                logger.info(f"Optimized prediction weights based on correlations: {weights}")
                logger.info(f"Weight optimization used correlations: {valid_correlations}")
                return weights
            else:
                logger.warning("No valid correlations found for weight optimization")
                return {
                    'season_avg': 0.4,
                    'recent_performance': 0.35,
                    'ballpark_factor': 0.15,
                    'weather_impact': 0.05,
                    'matchup_advantage': 0.05
                }
            
        except Exception as e:
            logger.error(f"Error optimizing prediction weights: {e}")
            logger.exception("Full traceback:")
            return {
                'season_avg': 0.4,
                'recent_performance': 0.35,
                'ballpark_factor': 0.15,
                'weather_impact': 0.05,
                'matchup_advantage': 0.05
            }

    def get_weather_impact(self, weather_data, ballpark_name=None):
        """Calculate weather impacts using calculated thresholds and enhanced validation"""
        if not weather_data:
            return {
                'factor': 1.0,
                'conditions': []
            }
        
        # Validate weather data quality
        validated_weather = self._validate_weather_data(weather_data)
        if not validated_weather:
            logger.warning("Invalid weather data, using neutral impact")
            return {
                'factor': 1.0,
                'conditions': ['Invalid weather data - using neutral impact']
            }
            
        # Check Redis cache first
        cache_key = f"weather_impact:{json.dumps(validated_weather)}:{ballpark_name or 'unknown'}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.debug("Redis cache hit for weather impact")
            return json.loads(cached_data)
        
        impact = {
            'factor': 1.0,
            'conditions': []
        }
        
        # Temperature impact using calculated thresholds
        temp = validated_weather['temperature']
        if temp > self.weather_thresholds['high_temp']:
            # Enhanced temperature boost - research shows hitting improves significantly in hot weather
            temp_boost = 1.05 + min(0.02, (temp - self.weather_thresholds['high_temp']) * 0.001)
            impact['factor'] *= temp_boost
            impact['conditions'].append(f"Temperature {temp}°F above {self.weather_thresholds['high_temp']}°F threshold favors hitting (+{((temp_boost-1)*100):.1f}%)")
        elif temp < self.weather_thresholds['low_temp']:
            # Enhanced temperature penalty - cold weather significantly affects hitting
            temp_penalty = 0.95 - min(0.03, (self.weather_thresholds['low_temp'] - temp) * 0.001)
            impact['factor'] *= temp_penalty
            impact['conditions'].append(f"Temperature {temp}°F below {self.weather_thresholds['low_temp']}°F threshold reduces hitting ({((temp_penalty-1)*100):.1f}%)")
            
        # Wind speed impact using calculated threshold
        wind_speed = validated_weather['wind_speed']
        if wind_speed > self.weather_thresholds['wind_speed']:
            # Enhanced wind penalty based on speed
            wind_penalty = 0.98 - min(0.02, (wind_speed - self.weather_thresholds['wind_speed']) * 0.001)
            impact['factor'] *= wind_penalty
            impact['conditions'].append(f"Wind speed {wind_speed}mph above {self.weather_thresholds['wind_speed']}mph threshold may affect hitting ({((wind_penalty-1)*100):.1f}%)")
        
        # Wind direction impact (enhanced feature)
        if ballpark_name and 'wind_direction_degrees' in validated_weather:
            wind_direction_degrees = validated_weather['wind_direction_degrees']
            if wind_direction_degrees > 0:  # Valid wind direction data
                wind_factor, wind_description = self.get_wind_direction_impact(wind_direction_degrees, ballpark_name)
                impact['factor'] *= wind_factor
                impact['conditions'].append(wind_description)
            
        # Precipitation impact using calculated factor
        conditions_lower = validated_weather['conditions'].lower()
        precipitation_factor = self.weather_thresholds.get('precipitation_factor', 0.97)
        
        if 'rain' in conditions_lower:
            impact['factor'] *= precipitation_factor
            rain_penalty = (1 - precipitation_factor) * 100
            impact['conditions'].append(f'Rain reduces hitting effectiveness (-{rain_penalty:.1f}%)')
        elif 'snow' in conditions_lower:
            # Snow has even more impact than rain
            snow_factor = max(0.90, precipitation_factor - 0.03)
            impact['factor'] *= snow_factor
            snow_penalty = (1 - snow_factor) * 100
            impact['conditions'].append(f'Snow significantly reduces hitting effectiveness (-{snow_penalty:.1f}%)')
        elif 'storm' in conditions_lower or 'thunderstorm' in conditions_lower:
            # Storms have severe impact
            storm_factor = max(0.85, precipitation_factor - 0.05)
            impact['factor'] *= storm_factor
            storm_penalty = (1 - storm_factor) * 100
            impact['conditions'].append(f'Storm conditions severely impact hitting (-{storm_penalty:.1f}%)')
        
        # Indoor stadium adjustments
        indoor_stadiums = {
            'Tropicana Field', 'Rogers Centre', 'Chase Field', 'LoanDepot Park',
            'Daikin Park', 'American Family Field', 'T-Mobile Park', 'Globe Life Field'
        }
        
        if ballpark_name in indoor_stadiums:
            # Reduce weather impact for domed/retractable roof stadiums
            original_factor = impact['factor']
            weather_effect = abs(impact['factor'] - 1.0)
            reduced_effect = weather_effect * 0.3  # Reduce weather impact by 70%
            impact['factor'] = 1.0 + (reduced_effect if impact['factor'] > 1.0 else -reduced_effect)
            
            if original_factor != impact['factor']:
                impact['conditions'].append(f'Indoor stadium reduces weather impact (adjusted from {((original_factor-1)*100):+.1f}% to {((impact['factor']-1)*100):+.1f}%)')
        
        # Cache for 3 hours since weather impacts are time-sensitive
        redis_client.setex(cache_key, 60 * 60 * 3, json.dumps(impact))
        return impact

    def _validate_weather_data(self, weather_data):
        """Validate weather data quality and reasonableness"""
        try:
            if not isinstance(weather_data, dict):
                return None
            
            # Check required fields
            required_fields = ['temperature', 'wind_speed', 'conditions']
            for field in required_fields:
                if field not in weather_data:
                    logger.warning(f"Missing required weather field: {field}")
                    return None
            
            # Validate temperature (reasonable range for baseball)
            temp = weather_data['temperature']
            if not isinstance(temp, (int, float)) or temp < 20 or temp > 120:
                logger.warning(f"Invalid temperature: {temp}°F")
                return None
            
            # Validate wind speed
            wind_speed = weather_data['wind_speed']
            if not isinstance(wind_speed, (int, float)) or wind_speed < 0 or wind_speed > 100:
                logger.warning(f"Invalid wind speed: {wind_speed}mph")
                return None
            
            # Validate conditions
            conditions = weather_data['conditions']
            if not isinstance(conditions, str) or len(conditions.strip()) == 0:
                logger.warning("Invalid weather conditions")
                conditions = "Unknown"
            
            # Validate wind direction if present
            wind_direction_degrees = weather_data.get('wind_direction_degrees', 0)
            if wind_direction_degrees and (not isinstance(wind_direction_degrees, (int, float)) or 
                                         wind_direction_degrees < 0 or wind_direction_degrees >= 360):
                logger.warning(f"Invalid wind direction: {wind_direction_degrees}°")
                wind_direction_degrees = 0
            
            # Return validated data
            validated = {
                'temperature': float(temp),
                'wind_speed': float(wind_speed),
                'conditions': str(conditions).strip(),
                'wind_direction_degrees': float(wind_direction_degrees) if wind_direction_degrees else 0
            }
            
            # Copy other fields that might be present
            for key, value in weather_data.items():
                if key not in validated:
                    validated[key] = value
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating weather data: {e}")
            return None

    async def get_weather(self, location):
        """Get weather data for a location"""
        try:
            # Check Redis cache first - removed date from cache key for more dynamic weather
            cache_key = f"weather:{location}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for weather in {location}")
                return json.loads(cached_data)

            if not self.weather_client:
                self.weather_client = python_weather.Client(unit=python_weather.IMPERIAL)
            weather = await self.weather_client.get(location)
            current = weather.current
            result = {
                'temperature': current.temperature,
                'wind_speed': current.wind_speed,
                'wind_direction': current.wind_direction.name if hasattr(current, 'wind_direction') and current.wind_direction else 'UNKNOWN',
                'wind_direction_degrees': current.wind_direction._WindDirection__degrees if hasattr(current, 'wind_direction') and current.wind_direction and hasattr(current.wind_direction, '_WindDirection__degrees') else 0,
                'humidity': current.humidity,
                'conditions': current.description
            }
            
            # Cache for 1 hour since weather changes frequently
            redis_client.setex(cache_key, 60 * 60, json.dumps(result))
            return result
        except Exception as e:
            print(f"Weather data unavailable for {location}: {e}")
            return None

    def get_player_name(self, player_id):
        """Get player's full name"""
        try:
            cache_key = f"player_name:{player_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for player {player_id} name")
                return cached_data
            player_info = self.mlb.get_player_info(player_id)
            if not player_info:
                logger.warning(f"No info found for player {player_id}")
                return "Unknown Player"
            first_name = player_info.get('firstName', '')
            last_name = player_info.get('lastName', '')
            full_name = f"{first_name} {last_name}".strip()
            if not full_name or full_name == '':
                logger.warning(f"Empty name for player {player_id}")
                return "Unknown Player"
            redis_client.setex(cache_key, CACHE_EXPIRATION, full_name)
            return full_name
        except Exception as e:
            logger.error(f"Error getting name for player {player_id}: {e}")
            return "Unknown Player"

    def get_todays_games(self):
        """Get regular season games for the prediction date (excluding Spring Training)"""
        date_str = self.prediction_date.strftime('%Y-%m-%d')
        schedule = self.mlb.get_schedule_by_date(date_str)
        
        # Filter to only regular season games
        if schedule:
            regular_season_games = [game for game in schedule if self.is_regular_season_game(game)]
            logger.info(f"Found {len(schedule)} total games, {len(regular_season_games)} regular season games on {date_str}")
            return regular_season_games
        
        return schedule
    
    def get_player_season_stats(self, player_id):
        """Get season stats for a player with enhanced Retrosheet integration for historical data"""
        try:
            # Check Redis cache first
            cache_key = f"player_stats:{player_id}:{self.current_season}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for player {player_id} stats")
                return pd.read_json(StringIO(cached_data))
            
            # Get player stats from MLB Stats API
            stats = self.mlb.get_player_stats(player_id, self.current_season)
            
            # If no stats for current season, try previous season
            if stats is None or not isinstance(stats, pd.DataFrame) or stats.empty:
                logger.info(f"No {self.current_season} stats for player {player_id}, trying {self.current_season - 1}")
                stats = self.mlb.get_player_stats(player_id, self.current_season - 1)
                
            if stats is None or not isinstance(stats, pd.DataFrame) or stats.empty:
                logger.warning(f"No stats found for player {player_id}")
                return pd.DataFrame()
            
            # Store only raw stats - no derived calculations
            # Cache the results
            redis_client.setex(cache_key, CACHE_EXPIRATION, stats.to_json())
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats for player {player_id}: {e}")
            return pd.DataFrame()

    def get_pitcher_season_stats(self, pitcher_id):
        """Get season stats for a pitcher"""
        try:
            # Check Redis cache first
            cache_key = f"pitcher_stats:{pitcher_id}:{self.current_season}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for pitcher {pitcher_id} stats")
                return pd.read_json(StringIO(cached_data))
            
            # Get pitcher stats from MLB Stats API
            stats = self.mlb.get_player_stats(pitcher_id, self.current_season, group='pitching')
            
            # If no stats for current season, try previous season
            if stats is None or not isinstance(stats, pd.DataFrame) or stats.empty:
                logger.info(f"No {self.current_season} stats for pitcher {pitcher_id}, trying {self.current_season - 1}")
                stats = self.mlb.get_player_stats(pitcher_id, self.current_season - 1, group='pitching')
                
            if stats is None or not isinstance(stats, pd.DataFrame) or stats.empty:
                logger.warning(f"No stats found for pitcher {pitcher_id}")
                return pd.DataFrame()
            
            # Store only raw stats - no derived calculations
            # Cache the results
            redis_client.setex(cache_key, CACHE_EXPIRATION, stats.to_json())
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats for pitcher {pitcher_id}: {e}")
            return pd.DataFrame()

    def get_historical_matchup(self, batter_id, pitcher_id):
        """Get historical matchup data between batter and pitcher using only Retrosheet data"""
        try:
            # Check Redis cache first
            cache_key = f"historical_matchup:{batter_id}:{pitcher_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                cached_result = json.loads(cached_data)
                # Verify cached data is valid
                if cached_result.get('plate_appearances', 0) > 0 or cached_result.get('avg', 0) > 0:
                    logger.info(f"Redis cache hit for matchup {batter_id} vs {pitcher_id}")
                    return cached_result
                else:
                    logger.debug(f"Cached matchup data appears empty for {batter_id} vs {pitcher_id}, refreshing")
            
            # Try Retrosheet data only - no MLB Stats API fallback
            if self.retrosheet_integration:
                try:
                    # Get player names for Retrosheet lookup
                    batter_name = self.get_player_name(batter_id)
                    pitcher_name = self.get_player_name(pitcher_id)
                    
                    if batter_name and pitcher_name:
                        logger.info(f"Attempting Retrosheet lookup for {batter_name} vs {pitcher_name}")
                        retrosheet_data = self.retrosheet_integration.get_historical_matchup_enhanced(
                            batter_id, pitcher_id, batter_name, pitcher_name
                        )
                        
                        # Check if we got meaningful Retrosheet data
                        if (retrosheet_data and 
                            retrosheet_data.get('retrosheet_available', False) and 
                            retrosheet_data.get('plate_appearances', 0) > 0):
                            
                            logger.info(f"Found Retrosheet matchup data: {batter_name} vs {pitcher_name} - "
                                      f"{retrosheet_data['hits']}/{retrosheet_data['plate_appearances']} "
                                      f"({retrosheet_data['avg']:.3f}) from {retrosheet_data['source']}")
                            
                            # Format for consistency with existing system
                            result = {
                                'avg': retrosheet_data['avg'],
                                'plate_appearances': retrosheet_data['plate_appearances'],
                                'hits': retrosheet_data['hits'],
                                'source': 'retrosheet',
                                'sample_size': retrosheet_data.get('sample_size', 'unknown'),
                                'singles': retrosheet_data.get('singles', 0),
                                'doubles': retrosheet_data.get('doubles', 0),
                                'triples': retrosheet_data.get('triples', 0),
                                'home_runs': retrosheet_data.get('home_runs', 0),
                                'rbi': retrosheet_data.get('rbi', 0)
                            }
                            
                            # Cache the results with longer expiration for Retrosheet data
                            redis_client.setex(cache_key, CACHE_EXPIRATION * 2, json.dumps(result))
                            return result
                        else:
                            logger.info(f"No Retrosheet data found for {batter_name} vs {pitcher_name}")
                    else:
                        logger.warning(f"Could not get player names for Retrosheet lookup: batter_id={batter_id}, pitcher_id={pitcher_id}")
                        
                except Exception as e:
                    logger.warning(f"Error querying Retrosheet data for {batter_id} vs {pitcher_id}: {e}")
            
            # No MLB Stats API fallback - return empty result if no Retrosheet data
            logger.info(f"No matchup data available for {batter_id} vs {pitcher_id} - Retrosheet only, no MLB Stats API fallback")
            result = {'avg': 0, 'plate_appearances': 0, 'hits': 0, 'source': 'no_data_available'}
            
            # Cache the empty result to avoid repeated lookups
            redis_client.setex(cache_key, CACHE_EXPIRATION // 2, json.dumps(result))  # Shorter cache for empty results
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical matchup for {batter_id} vs {pitcher_id}: {e}")
            # Return empty result and cache it briefly to avoid repeated failures
            result = {'avg': 0, 'plate_appearances': 0, 'hits': 0, 'source': 'error'}
            try:
                redis_client.setex(cache_key, 300, json.dumps(result))  # 5 minute cache for errors
            except:
                pass
            return result
    
    def get_recent_performance(self, player_id, days_back=30):
        """Get player's recent performance using statcast_batter for accurate last 30 days stats"""
        try:
            # Check Redis cache first
            cache_key = f"recent_performance:{player_id}:{days_back}:{self.current_season}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for player {player_id} recent performance")
                return json.loads(cached_data)

            # Calculate date range - ensure we don't go back into previous season
            end_date = self.prediction_date
            start_date = end_date - timedelta(days=days_back)
            
            # Get the actual season start date (includes Spring Training)
            season_start = self.get_season_start_date()
            
            # Don't go back further than the current season start
            if start_date < season_start:
                start_date = season_start
                logger.info(f"[get_recent_performance] Limited start date to season start: {season_start.strftime('%Y-%m-%d')} for player {player_id}")

            # Use statcast_batter for recent stats
            try:
                logger.info(f"[get_recent_performance] Calling statcast_batter for player_id={player_id}, start={start_date.strftime('%Y-%m-%d')}, end={end_date.strftime('%Y-%m-%d')}")
                data = statcast_batter(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), player_id)
                logger.info(f"[get_recent_performance] statcast_batter returned {0 if data is None else len(data)} rows for player_id={player_id}")
            except Exception as e:
                logger.error(f"Error fetching statcast_batter data for player {player_id}: {e}")
                data = None

            if data is None or data.empty:
                logger.warning(f"No recent stats found for player {player_id}")
                result = {'recent_avg': 0, 'hits': 0, 'plate_appearances': 0}
                redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
                return result

            # Calculate hits and plate appearances correctly
            hits = data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
            
            # Plate appearances: count all events that constitute a completed plate appearance
            completed_pa_events = [
                'single', 'double', 'triple', 'home_run',  # hits
                'strikeout', 'field_out', 'force_out', 'grounded_into_double_play', 
                'fielders_choice', 'fielders_choice_out', 'double_play',  # outs
                'walk', 'intentional_walk', 'hit_by_pitch',  # free passes
                'sac_fly', 'sac_bunt', 'catcher_interf'  # sacrifices and interference
            ]
            plate_appearances = data['events'].isin(completed_pa_events).sum()
            
            recent_avg = hits / plate_appearances if plate_appearances > 0 else 0

            result = {
                'recent_avg': recent_avg,
                'hits': int(hits),
                'plate_appearances': int(plate_appearances)
            }
            redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
            return result
        except Exception as e:
            logger.error(f"Error getting recent performance for player {player_id}: {e}")
            return {'recent_avg': 0, 'hits': 0, 'plate_appearances': 0}

    def calculate_weather_factor(self, weather_data):
        """DEPRECATED: Use get_weather_impact() instead for enhanced weather calculations"""
        # This method has been replaced by get_weather_impact() which provides:
        # - Enhanced temperature effects based on calculated thresholds
        # - Wind direction analysis based on ballpark orientation  
        # - Precipitation factor calculations from historical data
        # - Indoor stadium adjustments
        # - Better weather data validation
        logger.warning("calculate_weather_factor() is deprecated. Use get_weather_impact() instead.")
        
        # For backward compatibility, use simplified calculation
        if not weather_data:
            return 1.0
        
        # Use the enhanced method instead
        impact = self.get_weather_impact(weather_data)
        return impact['factor']

    def calculate_matchup_advantage(self, batter_stats, pitcher_stats, historical_matchup=None, batter_id=None, pitcher_id=None):
        """Calculate batter vs pitcher matchup advantage using platoon splits"""
        # Calculate averages on-the-fly from raw stats
        if not batter_stats.empty and 'hits' in batter_stats.columns and 'plateAppearances' in batter_stats.columns:
            batter_avg = batter_stats['hits'].iloc[0] / batter_stats['plateAppearances'].iloc[0] if batter_stats['plateAppearances'].iloc[0] > 0 else 0
        else:
            batter_avg = 0
            
        if not pitcher_stats.empty and 'hits' in pitcher_stats.columns and 'plateAppearances' in pitcher_stats.columns:
            pitcher_avg_against = pitcher_stats['hits'].iloc[0] / pitcher_stats['plateAppearances'].iloc[0] if pitcher_stats['plateAppearances'].iloc[0] > 0 else 0
        else:
            pitcher_avg_against = 0

        # Get handedness information
        try:
            # Use batter_id and pitcher_id if provided, else fallback to None
            batter_info = self.mlb.get_player_info(batter_id) if batter_id else {}
            pitcher_info = self.mlb.get_player_info(pitcher_id) if pitcher_id else {}

            batter_side = batter_info.get('batSide', {}).get('code', 'R')
            pitcher_side = pitcher_info.get('pitchHand', {}).get('code', 'R')

            # Map to platoon key
            matchup_key = f"{batter_side}_vs_{pitcher_side}"
            # Convert to expected keys (L_vs_R, L_vs_L, etc.)
            matchup_key = matchup_key.replace('L_vs_R', 'L_vs_R').replace('L_vs_L', 'L_vs_L').replace('R_vs_R', 'R_vs_R').replace('R_vs_L', 'R_vs_L')
            platoon_avg = self.platoon_stats.get(matchup_key, 0.250)

            # Weight historical matchup more heavily if available
            if historical_matchup and historical_matchup['plate_appearances'] >= 10:
                matchup_score = (historical_matchup['avg'] * 0.4 +
                               platoon_avg * 0.3 +
                               ((batter_avg + (1 - pitcher_avg_against)) / 2) * 0.3)
            else:
                matchup_score = (platoon_avg * 0.4 +
                               ((batter_avg + (1 - pitcher_avg_against)) / 2) * 0.6)

            return matchup_score

        except Exception as e:
            print(f"Error calculating matchup advantage: {e}")
            # Fallback to basic calculation if platoon data unavailable
            return (batter_avg + (1 - pitcher_avg_against)) / 2

    async def predict_hit_probability(self, batter_id, pitcher_id=None, ballpark=None, weather_location=None, min_recent_pa=15, game_pk=None, game_date=None, log_prediction=True):
        """Calculate probability using optimized weights"""
        try:
            # Get season stats
            batter_stats = self.get_player_season_stats(batter_id)
            if batter_stats.empty:
                logger.warning(f"No season stats found for batter {batter_id}")
                return 0
            
            # Verify this is a position player, not a pitcher
            player_info = self.mlb.get_player_info(batter_id)
            logger.info(f"DEBUG: player_info for {batter_id}: {player_info}")
            if not player_info or player_info.get('primaryPosition', {}).get('code') in ['1', 'TWP']:
                logger.warning(f"Player {batter_id} is a pitcher, skipping")
                return 0
            
            # Base probability from season average - calculate on-the-fly
            if 'hits' in batter_stats.columns and 'plateAppearances' in batter_stats.columns:
                season_avg = batter_stats['hits'].iloc[0] / batter_stats['plateAppearances'].iloc[0] if batter_stats['plateAppearances'].iloc[0] > 0 else 0
            else:
                season_avg = 0
            
            # Get recent performance
            recent_perf = self.get_recent_performance(batter_id)
            
            # Check minimum recent plate appearances requirement
            if recent_perf['plate_appearances'] < min_recent_pa:
                logger.info(f"Player {batter_id} ({self.get_player_name(batter_id)}) has only {recent_perf['plate_appearances']} plate appearances in last 30 days (minimum required: {min_recent_pa}), skipping")
                return 0
            
            # Calculate initial probability using optimized weights
            base_probability = (
                season_avg * self.prediction_weights['season_avg'] +
                recent_perf['recent_avg'] * self.prediction_weights['recent_performance']
            )
            
            # Track factors for logging
            ballpark_factor = 1.0
            weather_factor = 1.0
            matchup_factor = 1.0
            
            # Adjust for matchup if available
            if pitcher_id:
                pitcher_stats = self.get_pitcher_season_stats(pitcher_id)
                historical_matchup = self.get_historical_matchup(batter_id, pitcher_id)
                
                # Get handedness information
                batter_side = player_info.get('batSide', {}).get('code', 'R')
                pitcher_info = self.mlb.get_player_info(pitcher_id)
                pitcher_side = pitcher_info.get('pitchHand', {}).get('code', 'R')
                
                # Get platoon advantage
                matchup_key = f"{batter_side}_vs_{pitcher_side}"
                platoon_avg = self.platoon_stats.get(matchup_key, 0.250)
                
                # Calculate pitcher BAA on-the-fly from raw stats
                if not pitcher_stats.empty and 'hits' in pitcher_stats.columns and 'plateAppearances' in pitcher_stats.columns:
                    pitcher_baa = pitcher_stats['hits'].iloc[0] / pitcher_stats['plateAppearances'].iloc[0] if pitcher_stats['plateAppearances'].iloc[0] > 0 else 0
                else:
                    pitcher_baa = 0
                
                # Calculate matchup score
                if historical_matchup and historical_matchup['plate_appearances'] >= 10:
                    matchup_score = (
                        historical_matchup['avg'] * 0.4 +
                        platoon_avg * 0.3 +
                        ((season_avg + (1 - pitcher_baa)) / 2) * 0.3
                    )
                else:
                    matchup_score = (
                        platoon_avg * 0.4 +
                        ((season_avg + (1 - pitcher_baa)) / 2) * 0.6
                    )
                
                # Calculate matchup factor
                matchup_factor = matchup_score / ((season_avg + recent_perf['recent_avg']) / 2) if (season_avg + recent_perf['recent_avg']) > 0 else 1.0
                
                base_probability = (
                    base_probability * (1 - self.prediction_weights['matchup_advantage']) +
                    matchup_score * self.prediction_weights['matchup_advantage']
                )
            
            # Apply ballpark factor
            if ballpark and ballpark in self.ballpark_factors:
                ballpark_factor = self.ballpark_factors[ballpark]
                base_probability *= ballpark_factor
            
            # Apply weather factor
            if weather_location:
                weather_data = await self.get_weather(weather_location)
                if weather_data:
                    weather_impact = self.get_weather_impact(weather_data, ballpark)
                    weather_factor = weather_impact['factor']
                    base_probability *= weather_factor
            
            final_probability = round(base_probability, 3)
            
            # Log prediction if requested and we have game context
            if log_prediction and game_pk and game_date:
                try:
                    player_name = self.get_player_name(batter_id)
                    self.log_prediction(
                        player_id=batter_id,
                        player_name=player_name,
                        game_date=game_date,
                        game_pk=game_pk,
                        ballpark=ballpark,
                        opposing_pitcher_id=pitcher_id,
                        season_stats=batter_stats,
                        recent_stats=recent_perf,
                        predicted_prob=final_probability,
                        ballpark_factor=ballpark_factor,
                        weather_factor=weather_factor,
                        matchup_factor=matchup_factor
                    )
                except Exception as e:
                    logger.error(f"Error logging prediction: {e}")
            
            return final_probability
            
        except Exception as e:
            logger.error(f"Error calculating hit probability for batter {batter_id}: {e}")
            return 0

    def get_team_roster(self, team_id):
        """Get active roster for a team"""
        try:
            # Check Redis cache first - removed date from cache key
            cache_key = f"team_roster:{team_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for team roster {team_id}")
                return json.loads(cached_data)

            roster = self.mlb.get_team_roster(team_id)
            # Filter for position players (non-pitchers)
            position_players = [
                player for player in roster
                if player.get('position', {}).get('code') not in ['1', 'TWP']
            ]
            
            # Cache for 24 hours since rosters don't change frequently
            redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(position_players))
            return position_players
        except Exception as e:
            print(f"Error getting roster for team {team_id}: {e}")
            return []

    def get_recent_lineup(self, team_id, days_back=3):
        """Get most recent lineup used by team"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Check Redis cache first
            cache_key = f"recent_lineup:{team_id}:{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for recent lineup: {team_id}")
                return json.loads(cached_data)
            
            # Get team's recent games
            schedule = self.mlb.get_team_schedule(
                team_id,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Look for most recent game with available lineup
            for game in reversed(schedule):
                try:
                    lineup = self.mlb.get_lineups(game['gamePk'])
                    if lineup:
                        # Return the lineup for the team we're interested in
                        is_home = game['teams']['home']['team']['id'] == team_id
                        result = lineup['home'] if is_home else lineup['away']
                        
                        # Cache for 24 hours
                        redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(result))
                        return result
                except:
                    continue
                    
        except Exception as e:
            print(f"Error getting recent lineup for team {team_id}: {e}")
        return None

    def get_probable_lineup(self, team_id):
        """Get probable lineup for a team"""
        try:
            # Check Redis cache first - removed date from cache key
            cache_key = f"probable_lineup:{team_id}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis cache hit for team {team_id} probable lineup")
                return json.loads(cached_data)
            
            # Get team roster
            roster = self.get_team_roster(team_id)
            if not roster:
                logger.warning(f"No roster found for team {team_id}")
                return []
            
            # Filter for position players (non-pitchers) and extract player IDs
            player_ids = []
            for player in roster:
                # Get player ID from the person object
                player_id = player.get('person', {}).get('id')
                if player_id:
                    player_ids.append(player_id)
            
            # Sort by jersey number if available, otherwise keep original order
            def get_sort_key(player):
                jersey_num = player.get('jerseyNumber')
                if jersey_num and jersey_num.isdigit():
                    return int(jersey_num)
                return 999  # Put non-numeric jersey numbers at the end
            
            # Sort roster by jersey number to get a reasonable batting order
            sorted_roster = sorted(roster, key=get_sort_key)
            
            # Extract player IDs from sorted roster
            player_ids = []
            for player in sorted_roster:
                player_id = player.get('person', {}).get('id')
                if player_id:
                    player_ids.append(player_id)
            
            # Cache the results
            redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(player_ids))
            return player_ids
            
        except Exception as e:
            logger.error(f"Error getting probable lineup for team {team_id}: {e}")
            return []

    def load_venue_locations(self):
        """Load venue location mapping from file"""
        try:
            with open('venue_locations.json', 'r') as f:
                venue_locations = json.load(f)
                # Convert string keys to integers
                return {int(k): v for k, v in venue_locations.items()}
        except FileNotFoundError:
            logger.warning("venue_locations.json not found, weather locations will not be available")
            return {}
        except Exception as e:
            logger.error(f"Error loading venue locations: {e}")
            return {}

    def get_weather_location_for_venue(self, venue_id, venue_name=None):
        """Get weather location for a venue ID"""
        if not hasattr(self, 'venue_locations'):
            self.venue_locations = self.load_venue_locations()
        
        if venue_id in self.venue_locations:
            return self.venue_locations[venue_id]['weather_location']
        
        # Fallback: try to map by name if venue_name is provided
        if venue_name:
            for venue_data in self.venue_locations.values():
                if venue_data['name'] == venue_name:
                    return venue_data['weather_location']
        
        logger.warning(f"No weather location found for venue ID {venue_id} ({venue_name})")
        return None

    async def get_todays_hit_probabilities(self, min_recent_pa=15, min_season_pa=0):
        """Get hit probabilities for games on prediction date"""
        try:
            # Get schedule for prediction date
            schedule = self.mlb.get_schedule_by_date(self.prediction_date.strftime('%Y-%m-%d'))
            if not schedule:
                logger.warning(f"No games scheduled for {self.prediction_date.strftime('%Y-%m-%d')}")
                return []
            
            probabilities = []
            for game in schedule:
                try:
                    # Skip completed games
                    game_status = game.get('status', {}).get('detailedState', '')
                    if game_status in ['Final', 'Game Over', 'Postponed', 'Cancelled']:
                        logger.info(f"Skipping {game_status} game: {game.get('teams', {}).get('away', {}).get('team', {}).get('name')} @ {game.get('teams', {}).get('home', {}).get('team', {}).get('name')}")
                        continue
                    
                    # Get game info
                    game_pk = game.get('gamePk')
                    if not game_pk:
                        continue
                    
                    # Get teams
                    home_team = game.get('teams', {}).get('home', {}).get('team', {})
                    away_team = game.get('teams', {}).get('away', {}).get('team', {})
                    if not home_team or not away_team:
                        continue
                    
                    # Get ballpark
                    ballpark = game.get('venue', {}).get('name')
                    if not ballpark:
                        continue
                    
                    # Get weather location
                    weather_location = self.get_weather_location_for_venue(game.get('venue', {}).get('id'), ballpark)
                    
                    # Get probable pitchers
                    home_pitcher_id = game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('id')
                    away_pitcher_id = game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('id')
                    
                    # Get game time
                    game_time = game.get('gameDate')
                    if game_time:
                        try:
                            # Parse UTC time
                            utc_time = datetime.strptime(game_time, '%Y-%m-%dT%H:%M:%SZ')
                            
                            # Convert to Eastern Time
                            utc_timezone = pytz.UTC
                            eastern_timezone = pytz.timezone('US/Eastern')
                            
                            # Make UTC time timezone-aware and convert to ET
                            utc_aware = utc_timezone.localize(utc_time)
                            et_time = utc_aware.astimezone(eastern_timezone)
                            
                            # Format in 12-hour format
                            game_time = et_time.strftime('%I:%M %p ET')
                        except Exception as e:
                            logger.error(f"Error parsing game time {game_time}: {e}")
                            game_time = "TBD"
                    
                    # Get lineups
                    home_batters = game.get('teams', {}).get('home', {}).get('batters', [])
                    away_batters = game.get('teams', {}).get('away', {}).get('batters', [])
                    
                    # If no lineups available, use probable lineups from roster
                    if not home_batters:
                        logger.info(f"No lineup available for {home_team.get('name')}, using probable lineup from roster.")
                        home_batters = self.get_probable_lineup(home_team.get('id'))
                    if not away_batters:
                        logger.info(f"No lineup available for {away_team.get('name')}, using probable lineup from roster.")
                        away_batters = self.get_probable_lineup(away_team.get('id'))
                    
                    # Calculate probabilities for home team batters
                    for batter_id in home_batters:
                        if batter_id:  # Only process if we have a valid ID
                            prob = await self.predict_hit_probability(
                                batter_id,
                                away_pitcher_id,
                                ballpark,
                                weather_location,
                                min_recent_pa,
                                game_pk,
                                self.prediction_date.strftime('%Y-%m-%d')
                            )
                            if prob > 0:  # Only include if we got a valid probability
                                # Check for doubleheader advantage
                                doubleheader_info = self.detect_doubleheader_advantage(schedule, home_team.get('id'), batter_id)
                                adjusted_prob = prob * doubleheader_info['advantage_factor']
                                
                                probabilities.append({
                                    'player_id': batter_id,
                                    'player_name': self.get_player_name(batter_id),
                                    'probability': adjusted_prob,
                                    'base_probability': prob,
                                    'doubleheader_advantage': doubleheader_info['advantage_factor'],
                                    'doubleheader_explanation': doubleheader_info['explanation'],
                                    'game_info': f"{away_team.get('name')} @ {home_team.get('name')}",
                                    'game_time': game_time,
                                    'game_date': self.prediction_date.strftime('%Y-%m-%d'),
                                    'ballpark': ballpark,
                                    'team': home_team.get('name'),
                                    'opponent_pitcher': self.get_player_name(away_pitcher_id) if away_pitcher_id else 'TBD'
                                })
                    
                    # Calculate probabilities for away team batters
                    for batter_id in away_batters:
                        if batter_id:  # Only process if we have a valid ID
                            prob = await self.predict_hit_probability(
                                batter_id,
                                home_pitcher_id,
                                ballpark,
                                weather_location,
                                min_recent_pa,
                                game_pk,
                                self.prediction_date.strftime('%Y-%m-%d')
                            )
                            if prob > 0:  # Only include if we got a valid probability
                                # Check for doubleheader advantage
                                doubleheader_info = self.detect_doubleheader_advantage(schedule, away_team.get('id'), batter_id)
                                adjusted_prob = prob * doubleheader_info['advantage_factor']
                                
                                probabilities.append({
                                    'player_id': batter_id,
                                    'player_name': self.get_player_name(batter_id),
                                    'probability': adjusted_prob,
                                    'base_probability': prob,
                                    'doubleheader_advantage': doubleheader_info['advantage_factor'],
                                    'doubleheader_explanation': doubleheader_info['explanation'],
                                    'game_info': f"{away_team.get('name')} @ {home_team.get('name')}",
                                    'game_time': game_time,
                                    'game_date': self.prediction_date.strftime('%Y-%m-%d'),
                                    'ballpark': ballpark,
                                    'team': away_team.get('name'),
                                    'opponent_pitcher': self.get_player_name(home_pitcher_id) if home_pitcher_id else 'TBD'
                                })
                except Exception as e:
                    logger.error(f"Error processing game {game.get('gamePk', 'Unknown')}: {e}")
                    continue
            
            # Sort by probability and deduplicate players (keep highest probability game for each player)
            probabilities_sorted = sorted(probabilities, key=lambda x: x['probability'], reverse=True)
            
            # Deduplicate by player_id, keeping the highest probability game
            seen_players = set()
            deduplicated_probabilities = []
            
            for prob in probabilities_sorted:
                player_id = prob['player_id']
                if player_id not in seen_players:
                    seen_players.add(player_id)
                    deduplicated_probabilities.append(prob)
                else:
                    # Player already seen with higher probability, log the duplicate
                    logger.debug(f"Duplicate player {prob['player_name']} found with lower probability {prob['probability']:.3f} in game {prob['game_info']} at {prob['game_time']}, keeping higher probability game")
            
            logger.info(f"Deduplicated {len(probabilities)} total predictions to {len(deduplicated_probabilities)} unique players")
            return deduplicated_probabilities
            
        except Exception as e:
            logger.error(f"Error getting today's hit probabilities: {e}")
            return []

    def initialize_training_system(self):
        """Initialize the training system and optimize weights if sufficient data exists"""
        try:
            logger.info("Initializing training system...")
            
            # Collect recent completed games for training
            training_data = self.collect_training_data(days_back=14)
            
            # Validate training data
            is_valid, message = self.validate_training_data(training_data)
            logger.info(f"Training data validation: {message}")
            
            if is_valid and len(training_data) >= 20:
                # Optimize prediction weights
                optimized_weights = self.optimize_prediction_weights(training_data)
                self.prediction_weights = optimized_weights
                logger.info("Training system initialized successfully with optimized weights")
            else:
                logger.info("Training system initialized with default weights (insufficient data for optimization)")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error initializing training system: {e}")
            return []

    async def cleanup(self):
        """Cleanup resources"""
        if self.weather_client:
            await self.weather_client.close()
            self.weather_client = None

    def update_training_data(self, game_data):
        """Update training data with actual game results"""
        try:
            date_str = game_data['gameDate']
            cache_key = f"training_data:{date_str}"
            cached_data = redis_client.get(cache_key)
            
            if not cached_data:
                return
                
            training_data = json.loads(cached_data)
            
            # Update actual hits for each batter in the game
            for team in [game_data['homeTeam'], game_data['awayTeam']]:
                for player in team['players']:
                    batter_id = player['id']
                    actual_hits = player.get('hits', 0)
                    
                    # Find and update the corresponding training data entry
                    for entry in training_data:
                        if entry.get('batter_id') == batter_id:
                            entry['actual_hits'] = actual_hits
                            break
            
            # Cache the updated training data
            redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(training_data))
            
            # Re-optimize weights with updated data
            self.prediction_weights = self.optimize_prediction_weights(training_data)
            
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

    def process_game(self, game):
        """Process a single game and return hit probabilities"""
        try:
            game_pk = game['gamePk']
            game_date = game['gameDate']
            venue = game['venue']['name']
            
            # Get lineups
            home_lineup = self.mlb.get_lineups(game_pk)
            away_lineup = self.mlb.get_lineups(game_pk)
            
            if not home_lineup or not away_lineup:
                logger.warning(f"No lineups available for game {game_pk}")
                return None
            
            # Get weather data
            weather_location = f"{game['venue']['location']['city']}, {game['venue']['location']['state']}"
            weather_data = asyncio.run(self.get_weather(weather_location))
            
            # Process each team's lineup
            home_team = {
                'name': game['teams']['home']['team']['name'],
                'players': []
            }
            
            away_team = {
                'name': game['teams']['away']['team']['name'],
                'players': []
            }
            
            # Process home team lineup
            for player in home_lineup:
                try:
                    batter_id = player['person']['id']
                    batter_name = self.get_player_name(batter_id)
                    
                    # Get pitcher if available
                    pitcher_id = None
                    if 'probablePitcher' in game['teams']['away']:
                        pitcher_id = game['teams']['away']['probablePitcher']['id']
                    
                    # Calculate hit probability
                    prob = asyncio.run(self.predict_hit_probability(
                        batter_id,
                        pitcher_id,
                        venue,
                        weather_location,
                        self.minimum_recent_pa,
                        game_pk,
                        game_date
                    ))
                    
                    home_team['players'].append({
                        'id': batter_id,
                        'name': batter_name,
                        'hit_probability': prob
                    })
                except Exception as e:
                    logger.error(f"Error processing home team player {player.get('person', {}).get('id')}: {e}")
                    continue
            
            # Process away team lineup
            for player in away_lineup:
                try:
                    batter_id = player['person']['id']
                    batter_name = self.get_player_name(batter_id)
                    
                    # Get pitcher if available
                    pitcher_id = None
                    if 'probablePitcher' in game['teams']['home']:
                        pitcher_id = game['teams']['home']['probablePitcher']['id']
                    
                    # Calculate hit probability
                    prob = asyncio.run(self.predict_hit_probability(
                        batter_id,
                        pitcher_id,
                        venue,
                        weather_location,
                        self.minimum_recent_pa,
                        game_pk,
                        game_date
                    ))
                    
                    away_team['players'].append({
                        'id': batter_id,
                        'name': batter_name,
                        'hit_probability': prob
                    })
                except Exception as e:
                    logger.error(f"Error processing away team player {player.get('person', {}).get('id')}: {e}")
                    continue
            
            # Update training data with game results
            game_data = {
                'gameDate': game_date,
                'homeTeam': home_team,
                'awayTeam': away_team
            }
            self.update_training_data(game_data)
            
            return {
                'game_pk': game_pk,
                'date': game_date,
                'venue': venue,
                'weather': weather_data,
                'home_team': home_team,
                'away_team': away_team
            }
            
        except Exception as e:
            logger.error(f"Error processing game {game.get('gamePk')}: {e}")
            return None

    def gather_training_data(self, days_back=30):
        """Gather historical game data for training"""
        try:
            training_data = []
            end_date = self.prediction_date
            start_date = end_date - timedelta(days=days_back)
            
            # Get the actual season start date (includes Spring Training)
            season_start = self.get_season_start_date()
            
            # Don't go back further than the current season start
            if start_date < season_start:
                start_date = season_start
                logger.info(f"[gather_training_data] Limited start date to season start: {season_start.strftime('%Y-%m-%d')}")
            
            # Check Redis cache first - include season in cache key
            cache_key = f"training_data:days_back_{days_back}:{self.current_season}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Redis cache hit for training data")
                return json.loads(cached_data)
            
            games_processed = 0
            games_with_lineups = 0
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                games = self.mlb.get_schedule_by_date(date_str)
                
                for game in games:
                    try:
                        games_processed += 1
                        
                        # Get lineups for the game - handle missing lineups gracefully
                        lineups = self.mlb.get_lineups(game['gamePk'])
                        
                        if not lineups:
                            logger.debug(f"No lineups available for game {game['gamePk']} on {date_str}")
                            continue
                            
                        home_lineup = lineups.get('home', [])
                        away_lineup = lineups.get('away', [])
                        
                        if not home_lineup and not away_lineup:
                            logger.debug(f"Empty lineups for game {game['gamePk']} on {date_str}")
                            continue
                            
                        games_with_lineups += 1
                        
                        # Process each batter in the lineup
                        for team_lineup in [home_lineup, away_lineup]:
                            for batter in team_lineup:
                                try:
                                    batter_id = batter.get('person', {}).get('id')
                                    if not batter_id:
                                        continue
                                    
                                    # Get season stats
                                    season_stats = self.get_player_season_stats(batter_id)
                                    if season_stats.empty:
                                        continue
                                        
                                    # Get recent performance
                                    recent_perf = self.get_recent_performance(batter_id)
                                    
                                    # Get matchup stats if pitcher is known
                                    matchup_hits = 0
                                    if 'probablePitcher' in game:
                                        pitcher_id = game['probablePitcher']['id']
                                        matchup_stats = self.mlb.get_matchup_stats(batter_id, pitcher_id)
                                        if matchup_stats and not matchup_stats.empty:
                                            matchup_hits = matchup_stats.get('hits', 0)
                                    
                                    # Add to training data
                                    training_data.append({
                                        'season_hits': season_stats['hits'].iloc[0] if 'hits' in season_stats.columns else 0,
                                        'recent_hits': recent_perf['hits'],
                                        'matchup_hits': matchup_hits,
                                        'actual_hits': 0  # This will be updated when we get actual game results
                                    })
                                except Exception as e:
                                    logger.debug(f"Error processing batter {batter.get('person', {}).get('id')}: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.debug(f"Error processing game {game.get('gamePk', 'Unknown')}: {e}")
                        continue
                
                current_date += timedelta(days=1)
            
            logger.info(f"Training data: processed {games_processed} games, {games_with_lineups} had lineups, collected {len(training_data)} batter records")
            
            # Cache the training data
            redis_client.setex(cache_key, CACHE_EXPIRATION, json.dumps(training_data))
            return training_data
            
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return []

    def get_wind_direction_impact(self, wind_direction_degrees, ballpark_name):
        """Calculate wind direction impact based on ballpark orientation and altitude"""
        # Ballpark orientations (home plate to center field direction in degrees)
        # 0° = North, 90° = East, 180° = South, 270° = West
        ballpark_orientations = {
            # American League East
            'Fenway Park': 90,  # East (Green Monster in left field)
            'Yankee Stadium': 90,  # East (short right field)
            'Oriole Park at Camden Yards': 45,  # Northeast
            'Rogers Centre': 45,  # Northeast
            'Tropicana Field': 45,  # Northeast (dome - less wind impact)
            
            # American League Central
            'Guaranteed Rate Field': 45,  # Northeast
            'Progressive Field': 45,  # Northeast
            'Comerica Park': 90,  # East
            'Kauffman Stadium': 90,  # East
            'American Family Field': 45,  # Northeast (retractable roof)
            
            # American League West
            'Angel Stadium': 45,  # Northeast
            'Daikin Park': 90,  # East (retractable roof)
            'Oakland Coliseum': 90,  # East
            'T-Mobile Park': 45,  # Northeast (retractable roof)
            'Globe Life Field': 45,  # Northeast (retractable roof)
            
            # National League East
            'Truist Park': 135,  # Southeast
            'Citi Field': 45,  # Northeast
            'Citizens Bank Park': 90,  # East
            'Nationals Park': 135,  # Southeast
            'LoanDepot Park': 90,  # East (retractable roof)
            
            # National League Central
            'Wrigley Field': 90,  # East (wind patterns from Lake Michigan)
            'Great American Ball Park': 135,  # Southeast
            'PNC Park': 45,  # Northeast
            'Busch Stadium': 90,  # East
            
            # National League West
            'Chase Field': 90,  # East (retractable roof)
            'Coors Field': 180,  # South (high altitude affects wind)
            'Dodger Stadium': 45,  # Northeast
            'Petco Park': 270,  # West (ocean influence)
            'Oracle Park': 270,  # West (bay influence, wind from right field)
            'Target Field': 45,  # Northeast
        }
        
        # Ballpark altitudes in feet above sea level
        ballpark_altitudes = {
            # American League East
            'Fenway Park': 20,
            'Yankee Stadium': 55,
            'Oriole Park at Camden Yards': 54,
            'Rogers Centre': 348,
            'Tropicana Field': 15,
            
            # American League Central
            'Guaranteed Rate Field': 595,
            'Progressive Field': 660,
            'Comerica Park': 585,
            'Kauffman Stadium': 750,
            'American Family Field': 635,
            
            # American League West
            'Angel Stadium': 160,
            'Daikin Park': 65,  # Houston
            'Oakland Coliseum': 6,
            'T-Mobile Park': 134,
            'Globe Life Field': 551,  # Arlington, TX
            
            # National League East
            'Truist Park': 1050,  # Atlanta
            'Citi Field': 20,
            'Citizens Bank Park': 65,
            'Nationals Park': 56,
            'LoanDepot Park': 8,  # Miami
            
            # National League Central
            'Wrigley Field': 595,
            'Great American Ball Park': 550,
            'PNC Park': 730,
            'Busch Stadium': 465,
            
            # National League West
            'Chase Field': 1090,  # Phoenix
            'Coors Field': 5280,  # Mile High - Denver
            'Dodger Stadium': 340,  # Los Angeles
            'Petco Park': 62,  # San Diego
            'Oracle Park': 12,  # San Francisco
            'Target Field': 815,  # Minneapolis
        }
        
        if ballpark_name not in ballpark_orientations:
            return 1.0, "Unknown ballpark orientation"
        
        ballpark_direction = ballpark_orientations[ballpark_name]
        altitude = ballpark_altitudes.get(ballpark_name, 0)
        
        # Calculate relative wind direction (wind direction relative to ballpark orientation)
        relative_wind = (wind_direction_degrees - ballpark_direction) % 360
        
        # Base wind impact based on direction relative to ballpark
        # Tailwind (behind home plate, 135-225°): Favors hitting (especially home runs)
        # Headwind (from center field, 315-45°): Reduces hitting
        # Crosswind (from sides, 45-135° and 225-315°): Moderate impact
        
        base_factor = 1.0
        wind_type = ""
        
        if 135 <= relative_wind <= 225:  # Tailwind
            base_factor = 1.03
            wind_type = "tailwind"
        elif 315 <= relative_wind or relative_wind <= 45:  # Headwind
            base_factor = 0.97
            wind_type = "headwind"
        elif 45 < relative_wind < 135:  # Left field crosswind
            base_factor = 1.01
            wind_type = "left field crosswind"
        elif 225 < relative_wind < 315:  # Right field crosswind
            base_factor = 1.01
            wind_type = "right field crosswind"
        else:
            base_factor = 1.0
            wind_type = "neutral wind"
        
        # Apply altitude adjustments
        altitude_factor = self._calculate_altitude_adjustment(altitude, base_factor, wind_type)
        final_factor = base_factor * altitude_factor
        
        # Generate description with altitude consideration
        wind_direction_name = self._degrees_to_direction(wind_direction_degrees)
        
        if wind_type == "tailwind":
            if altitude > 3000:
                description = f"Tailwind from {wind_direction_name} favors hitting (enhanced by high altitude at {altitude}ft)"
            else:
                description = f"Tailwind from {wind_direction_name} favors hitting"
        elif wind_type == "headwind":
            if altitude > 3000:
                description = f"Headwind from {wind_direction_name} reduces hitting (partially offset by high altitude at {altitude}ft)"
            else:
                description = f"Headwind from {wind_direction_name} reduces hitting"
        elif wind_type == "left field crosswind":
            description = f"Left field crosswind from {wind_direction_name} slightly favors right-handed batters"
            if altitude > 3000:
                description += f" (enhanced by altitude at {altitude}ft)"
        elif wind_type == "right field crosswind":
            description = f"Right field crosswind from {wind_direction_name} slightly favors left-handed batters"
            if altitude > 3000:
                description += f" (enhanced by altitude at {altitude}ft)"
        else:
            if altitude > 3000:
                description = f"Wind from {wind_direction_name} has neutral directional impact but benefits from high altitude at {altitude}ft"
            else:
                description = f"Wind from {wind_direction_name} has neutral impact"
        
        return final_factor, description
    
    def _calculate_altitude_adjustment(self, altitude, base_factor, wind_type):
        """Calculate altitude adjustment factor for wind effects"""
        # Sea level baseline
        if altitude <= 500:
            return 1.0
        
        # Calculate altitude effect - air density decreases with altitude
        # At higher altitudes, wind effects are amplified due to thinner air
        # Formula based on barometric pressure changes with altitude
        
        # Standard atmosphere: pressure decreases ~1% per 300 feet
        pressure_ratio = (1 - (altitude * 0.0000225577)) ** 5.25588
        air_density_ratio = pressure_ratio
        
        # Altitude categories and their effects
        if altitude >= 5000:  # Mile-high and above (Coors Field)
            # Extreme altitude - significant amplification of all wind effects
            if wind_type in ["tailwind", "headwind"]:
                altitude_multiplier = 1.15  # 15% amplification
            else:  # crosswinds
                altitude_multiplier = 1.08  # 8% amplification
        elif altitude >= 3000:  # High altitude (Chase Field, Truist Park)
            # High altitude - moderate amplification
            if wind_type in ["tailwind", "headwind"]:
                altitude_multiplier = 1.08  # 8% amplification
            else:  # crosswinds
                altitude_multiplier = 1.04  # 4% amplification
        elif altitude >= 1500:  # Moderate altitude
            # Moderate altitude - slight amplification
            if wind_type in ["tailwind", "headwind"]:
                altitude_multiplier = 1.04  # 4% amplification
            else:  # crosswinds
                altitude_multiplier = 1.02  # 2% amplification
        elif altitude >= 750:  # Elevated (many Midwest stadiums)
            # Slightly elevated - minimal amplification
            if wind_type in ["tailwind", "headwind"]:
                altitude_multiplier = 1.02  # 2% amplification
            else:  # crosswinds
                altitude_multiplier = 1.01  # 1% amplification
        else:  # Low altitude (500-750 feet)
            altitude_multiplier = 1.0  # No significant effect
        
        return altitude_multiplier

    def _degrees_to_direction(self, degrees):
        """Convert wind direction degrees to compass direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]

    def detect_doubleheader_advantage(self, schedule, team_id, player_id):
        """Detect if a team has a regular season doubleheader and calculate advantage for the player"""
        try:
            doubleheader_games = []
            
            # Find all regular season games for this team on the prediction date
            for game in schedule:
                # Skip non-regular season games (Spring Training, etc.)
                if not self.is_regular_season_game(game):
                    logger.debug(f"Skipping non-regular season doubleheader game {game.get('gamePk')}")
                    continue
                    
                home_team_id = game.get('teams', {}).get('home', {}).get('team', {}).get('id')
                away_team_id = game.get('teams', {}).get('away', {}).get('team', {}).get('id')
                
                if team_id in [home_team_id, away_team_id]:
                    is_doubleheader = game.get('doubleHeader', 'N')
                    game_number = game.get('gameNumber', 1)
                    
                    # Check for both traditional doubleheaders ('Y') and split doubleheaders ('S')
                    if is_doubleheader in ['Y', 'S']:
                        doubleheader_games.append({
                            'game_pk': game.get('gamePk'),
                            'game_number': game_number,
                            'is_doubleheader': True,
                            'doubleheader_type': is_doubleheader  # 'Y' = traditional, 'S' = split
                        })
            
            # If no doubleheader, return no advantage
            if len(doubleheader_games) == 0:
                return {
                    'has_doubleheader': False,
                    'advantage_factor': 1.0,
                    'games_count': len(doubleheader_games),
                    'explanation': 'No doubleheader scheduled'
                }
            elif len(doubleheader_games) == 1:
                # Check if this is a split doubleheader
                game_info = doubleheader_games[0]
                if game_info.get('doubleheader_type') == 'S':
                    # Split doubleheader - player likely to play in this game with potential for more opportunities
                    doubleheader_advantage = True
                    explanation_suffix = f"(Split doubleheader - Game #{game_info['game_number']})"
                else:
                    # Traditional doubleheader but only one game found - unusual, treat as no advantage
                    return {
                        'has_doubleheader': False,
                        'advantage_factor': 1.0,
                        'games_count': len(doubleheader_games),
                        'explanation': 'Incomplete doubleheader data'
                    }
            else:
                # Multiple games found - traditional doubleheader
                doubleheader_advantage = True
                explanation_suffix = f"({len(doubleheader_games)} games)"
            
            # Calculate doubleheader advantage
            # Players get more opportunities to get hits in doubleheaders
            # Regular starters are more likely to play in both games
            
            # Get player position to determine if they're a regular starter
            player_info = self.mlb.get_player_info(player_id)
            if not player_info:
                return {
                    'has_doubleheader': True,
                    'advantage_factor': 1.0,
                    'games_count': len(doubleheader_games),
                    'explanation': 'Player info unavailable for doubleheader analysis'
                }
            
            position_code = player_info.get('primaryPosition', {}).get('code')
            
            # Determine likelihood of playing in both games based on position
            # Position players (except catchers) are more likely to play both games
            # Pitchers are less likely to play both games
            
            if position_code in ['1', 'TWP']:  # Pitcher
                # Pitchers rarely play in both games of a doubleheader
                advantage_factor = 1.0
                explanation = 'Pitcher unlikely to play in both doubleheader games'
            elif position_code == '2':  # Catcher
                # Catchers often get rest in one game of a doubleheader
                advantage_factor = 1.02
                explanation = 'Catcher may play in both doubleheader games (modest advantage)'
            else:  # Position players (3-9)
                # Regular position players are most likely to play in both games
                # Get season stats to determine if they're a regular starter
                season_stats = self.get_player_season_stats(player_id)
                
                if not season_stats.empty:
                    games_played = season_stats.get('gamesPlayed', pd.Series([0])).iloc[0]
                    # If they've played in most of the season, they're likely a regular
                    if games_played > 100:  # Regular starter
                        advantage_factor = 1.08  # 8% boost for regular starters
                        explanation = 'Regular starter likely to play in both doubleheader games (significant advantage)'
                    elif games_played > 50:  # Semi-regular
                        advantage_factor = 1.04  # 4% boost for semi-regulars
                        explanation = 'Semi-regular player may play in both doubleheader games (moderate advantage)'
                    else:  # Bench player
                        advantage_factor = 1.01  # 1% boost for bench players
                        explanation = 'Bench player might get opportunities in doubleheader (minimal advantage)'
                else:
                    # Default advantage for position players
                    advantage_factor = 1.04
                    explanation = 'Position player may play in both doubleheader games (moderate advantage)'
            
            return {
                'has_doubleheader': True,
                'advantage_factor': advantage_factor,
                'games_count': len(doubleheader_games),
                'explanation': explanation + explanation_suffix
            }
            
        except Exception as e:
            logger.error(f"Error detecting doubleheader advantage for player {player_id}, team {team_id}: {e}")
            return {
                'has_doubleheader': False,
                'advantage_factor': 1.0,
                'games_count': 0,
                'explanation': 'Error in doubleheader analysis'
            }

    def validate_training_data(self, training_data):
        """Validate training data quality and completeness with enhanced analysis"""
        if not training_data:
            return False, "No training data provided"
        
        if len(training_data) < 10:
            return False, f"Insufficient training data: {len(training_data)} records (need 10+)"
        
        # Check for required fields in enhanced format
        required_fields = ['player_id', 'game_date', 'season_avg', 'recent_avg', 'predicted_prob', 'actual_hits', 'got_hit']
        sample_record = training_data[0]
        missing_fields = [field for field in required_fields if field not in sample_record]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Enhanced validation metrics
        df = pd.DataFrame(training_data)
        
        # Count records with meaningful data
        valid_predictions = df['predicted_prob'].notna().sum()
        valid_outcomes = df['actual_hits'].notna().sum()
        
        # Check for data quality issues
        quality_issues = []
        
        # Check for too many identical predictions (indicates placeholder data)
        if 'predicted_prob' in df.columns:
            unique_predictions = df['predicted_prob'].nunique()
            if unique_predictions < max(3, len(training_data) * 0.1):
                quality_issues.append(f"Low prediction diversity: {unique_predictions} unique values")
        
        # Check prediction probability range
        if 'predicted_prob' in df.columns:
            min_prob = df['predicted_prob'].min()
            max_prob = df['predicted_prob'].max()
            if max_prob - min_prob < 0.05:
                quality_issues.append(f"Narrow prediction range: {min_prob:.3f} - {max_prob:.3f}")
        
        # Check for reasonable hit rates
        if 'got_hit' in df.columns:
            hit_rate = df['got_hit'].mean()
            if hit_rate < 0.1 or hit_rate > 0.6:
                quality_issues.append(f"Unusual hit rate: {hit_rate:.3f} (expected 0.15-0.35)")
        
        # Check for season stats variation
        if 'season_avg' in df.columns:
            season_var = df['season_avg'].var()
            if season_var < 0.001:
                quality_issues.append("No variation in season averages (possible placeholder data)")
        
        # Calculate training data effectiveness
        effectiveness_score = 0
        max_score = 100
        
        # Data completeness (25 points)
        completeness = (valid_predictions + valid_outcomes) / (2 * len(training_data))
        effectiveness_score += completeness * 25
        
        # Prediction diversity (25 points)
        if 'predicted_prob' in df.columns:
            diversity = min(1.0, df['predicted_prob'].nunique() / (len(training_data) * 0.3))
            effectiveness_score += diversity * 25
        
        # Reasonable outcome distribution (25 points)
        if 'got_hit' in df.columns:
            hit_rate = df['got_hit'].mean()
            # Score based on how close hit rate is to expected MLB average (~0.25)
            hit_rate_score = 1.0 - abs(hit_rate - 0.25) / 0.25
            effectiveness_score += max(0, hit_rate_score) * 25
        
        # Statistical validity (25 points)
        if len(quality_issues) == 0:
            effectiveness_score += 25
        else:
            effectiveness_score += max(0, 25 - len(quality_issues) * 5)
        
        # Determine overall status
        if effectiveness_score >= 80:
            status = "Excellent"
        elif effectiveness_score >= 60:
            status = "Good"
        elif effectiveness_score >= 40:
            status = "Fair"
        else:
            status = "Poor"
        
        # Build detailed message
        message_parts = [
            f"Training data quality: {status} ({effectiveness_score:.0f}/100)",
            f"{len(training_data)} total records, {valid_outcomes} with outcomes ({valid_outcomes/len(training_data):.1%})"
        ]
        
        if 'predicted_prob' in df.columns:
            message_parts.append(f"Prediction range: {df['predicted_prob'].min():.3f} - {df['predicted_prob'].max():.3f}")
        
        if 'got_hit' in df.columns:
            hit_rate = df['got_hit'].mean()
            message_parts.append(f"Hit rate: {hit_rate:.3f}")
        
        if quality_issues:
            message_parts.append(f"Quality issues: {'; '.join(quality_issues)}")
        
        message = " | ".join(message_parts)
        
        # Return validation result
        is_valid = effectiveness_score >= 40 and len(missing_fields) == 0
        return is_valid, message

    def collect_training_data(self, days_back=14):
        """Collect recent completed games for training data with enhanced prediction reconstruction"""
        try:
            training_data = []
            
            # For the current scenario where API statuses are not reliable,
            # we'll check for actual box score data to determine completion
            current_date = datetime.now()
            end_date = current_date - timedelta(days=1)  # Start from yesterday
            start_date = end_date - timedelta(days=days_back)
            
            # Don't go back further than season start
            season_start = datetime(self.current_season, 3, 1)
            if start_date < season_start:
                start_date = season_start
                logger.info(f"[collect_training_data] Limited start date to season start: {start_date.strftime('%Y-%m-%d')}")
            
            # Check cache first
            cache_key = f"training_data:hybrid_method:{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Using cached hybrid training data")
                return json.loads(cached_data)
            
            logger.info(f"Collecting hybrid training data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # First, try to get logged predictions for this period
            logged_predictions = self.get_logged_predictions(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            logger.info(f"Found {len(logged_predictions)} logged predictions for training period")
            
            # Collect actual outcomes from completed games
            outcomes_data = self._collect_game_outcomes(start_date, end_date)
            logger.info(f"Found {len(outcomes_data)} game outcome records")
            
            # If we have logged predictions, merge them with outcomes
            if logged_predictions and outcomes_data:
                merged_data = self.merge_predictions_with_outcomes(logged_predictions, outcomes_data)
                if merged_data:
                    logger.info(f"Successfully merged {len(merged_data)} logged predictions with outcomes")
                    # Cache the results for 6 hours
                    redis_client.setex(cache_key, 60 * 60 * 6, json.dumps(merged_data))
                    return merged_data
            
            # Fallback to enhanced reconstruction method if no logged predictions available
            logger.info("No logged predictions available, falling back to reconstruction method")
            
            games_checked = 0
            completed_games_found = 0
            records_collected = 0
            
            current_check_date = start_date
            while current_check_date <= end_date:
                date_str = current_check_date.strftime('%Y-%m-%d')
                games = self.mlb.get_schedule_by_date(date_str)
                
                if not games:
                    current_check_date += timedelta(days=1)
                    continue
                
                for game in games:
                    try:
                        games_checked += 1
                        game_pk = game.get('gamePk')
                        if not game_pk:
                            continue
                        
                        # Try to get box score - if it exists and has data, the game was completed
                        box_score = self.get_game_box_score(game_pk)
                        if not box_score:
                            continue  # No box score = game not completed
                        
                        # Check if box score has actual batting statistics
                        has_batting_stats = False
                        total_at_bats = 0
                        
                        for team_key in ['home', 'away']:
                            team_data = box_score.get('teams', {}).get(team_key, {})
                            players = team_data.get('players', {})
                            
                            for player_key, player_data in players.items():
                                if not player_key.startswith('ID'):
                                    continue
                                
                                stats = player_data.get('stats', {}).get('batting', {})
                                if stats and stats.get('atBats', 0) > 0:
                                    has_batting_stats = True
                                    total_at_bats += stats.get('atBats', 0)
                        
                        if not has_batting_stats or total_at_bats < 10:
                            continue  # Not enough batting data = game likely not completed properly
                        
                        completed_games_found += 1
                        logger.info(f"Processing completed game: {game_pk} on {date_str}")
                        
                        # Get game context for historical reconstruction
                        ballpark = game.get('venue', {}).get('name')
                        weather_location = self.get_weather_location_for_venue(
                            game.get('venue', {}).get('id'), 
                            ballpark
                        )
                        
                        # Extract player performances from box score with enhanced prediction data
                        for team_key in ['home', 'away']:
                            team_data = box_score.get('teams', {}).get(team_key, {})
                            players = team_data.get('players', {})
                            
                            # Get probable pitcher for this team's opponents
                            opposing_team_key = 'away' if team_key == 'home' else 'home'
                            probable_pitcher_id = game.get('teams', {}).get(opposing_team_key, {}).get('probablePitcher', {}).get('id')
                            
                            for player_key, player_data in players.items():
                                try:
                                    if not player_key.startswith('ID'):
                                        continue
                                    
                                    player_id = int(player_key[2:])  # Remove 'ID' prefix
                                    stats = player_data.get('stats', {}).get('batting', {})
                                    
                                    if not stats or stats.get('atBats', 0) == 0:
                                        continue  # Skip if didn't bat
                                    
                                    # Verify this is a position player, not a pitcher
                                    player_info = self.mlb.get_player_info(player_id)
                                    if not player_info or player_info.get('primaryPosition', {}).get('code') in ['1', 'TWP']:
                                        continue  # Skip pitchers
                                    
                                    hits = stats.get('hits', 0)
                                    at_bats = stats.get('atBats', 0)
                                    plate_appearances = stats.get('plateAppearances', at_bats)  # Fall back to at_bats if PA not available
                                    
                                    # Reconstruct historical stats as they would have been on game date
                                    game_date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                    historical_season_stats = self._get_historical_season_stats(player_id, game_date_obj)
                                    historical_recent_stats = self._get_historical_recent_performance(player_id, game_date_obj)
                                    
                                    # Calculate what the prediction would have been
                                    predicted_prob = self._calculate_historical_prediction(
                                        player_id, 
                                        probable_pitcher_id,
                                        ballpark,
                                        historical_season_stats,
                                        historical_recent_stats,
                                        game_date_obj
                                    )
                                    
                                    # Create enhanced training record
                                    training_record = {
                                        'player_id': int(player_id),
                                        'player_name': self.get_player_name(player_id),
                                        'game_date': date_str,
                                        'game_pk': int(game_pk),
                                        'ballpark': ballpark,
                                        'opposing_pitcher_id': int(probable_pitcher_id) if probable_pitcher_id else None,
                                        'season_avg': float(historical_season_stats.get('avg', 0)),
                                        'season_hits': int(historical_season_stats.get('hits', 0)),
                                        'season_pa': int(historical_season_stats.get('plate_appearances', 0)),
                                        'recent_avg': float(historical_recent_stats.get('recent_avg', 0)),
                                        'recent_hits': int(historical_recent_stats.get('hits', 0)),
                                        'recent_pa': int(historical_recent_stats.get('plate_appearances', 0)),
                                        'predicted_prob': float(predicted_prob),
                                        'actual_hits': int(hits),
                                        'actual_at_bats': int(at_bats),
                                        'actual_pa': int(plate_appearances),
                                        'got_hit': int(1 if hits > 0 else 0),
                                        'prediction_error': float(abs(predicted_prob - (1 if hits > 0 else 0))),
                                        'ballpark_factor': float(self.ballpark_factors.get(ballpark, 1.0) if ballpark else 1.0)
                                    }
                                    
                                    training_data.append(training_record)
                                    records_collected += 1
                                    
                                except Exception as e:
                                    logger.debug(f"Error processing player {player_key}: {e}")
                                    continue
                        
                        # Limit to avoid too much processing
                        if completed_games_found >= 20:
                            logger.info(f"Reached limit of {completed_games_found} completed games")
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error processing game {game.get('gamePk', 'Unknown')}: {e}")
                        continue
                
                if completed_games_found >= 20:
                    break
                    
                current_check_date += timedelta(days=1)
            
            logger.info(f"Enhanced training data collection: {games_checked} games checked, {completed_games_found} completed games found, {records_collected} player records")
            
            # Cache the results for 6 hours
            if len(training_data) > 0:
                redis_client.setex(cache_key, 60 * 60 * 6, json.dumps(training_data))
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting enhanced training data: {e}")
            return []

    def _collect_game_outcomes(self, start_date, end_date):
        """Collect game outcomes (hits, at-bats) for completed regular season games only"""
        try:
            outcomes = []
            current_check_date = start_date
            
            while current_check_date <= end_date:
                date_str = current_check_date.strftime('%Y-%m-%d')
                games = self.mlb.get_schedule_by_date(date_str)
                
                for game in games:
                    try:
                        # Skip non-regular season games (Spring Training, etc.)
                        if not self.is_regular_season_game(game):
                            logger.debug(f"Skipping non-regular season game {game.get('gamePk')} on {date_str}")
                            continue
                            
                        game_pk = game.get('gamePk')
                        if not game_pk:
                            continue
                        
                        box_score = self.get_game_box_score(game_pk)
                        if not box_score:
                            continue
                        
                        # Extract outcomes for all batters
                        for team_key in ['home', 'away']:
                            team_data = box_score.get('teams', {}).get(team_key, {})
                            players = team_data.get('players', {})
                            
                            for player_key, player_data in players.items():
                                if not player_key.startswith('ID'):
                                    continue
                                
                                player_id = int(player_key[2:])
                                stats = player_data.get('stats', {}).get('batting', {})
                                
                                if not stats or stats.get('atBats', 0) == 0:
                                    continue
                                
                                hits = stats.get('hits', 0)
                                at_bats = stats.get('atBats', 0)
                                plate_appearances = stats.get('plateAppearances', at_bats)
                                
                                outcome_record = {
                                    'player_id': int(player_id),
                                    'game_pk': int(game_pk),
                                    'game_date': date_str,
                                    'actual_hits': int(hits),
                                    'actual_at_bats': int(at_bats),
                                    'actual_pa': int(plate_appearances),
                                    'got_hit': int(1 if hits > 0 else 0)
                                }
                                
                                outcomes.append(outcome_record)
                    
                    except Exception as e:
                        logger.debug(f"Error processing game outcomes for {game.get('gamePk')}: {e}")
                        continue
                
                current_check_date += timedelta(days=1)
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error collecting game outcomes: {e}")
            return []

    def _get_historical_season_stats(self, player_id, as_of_date):
        """Get season stats as they would have been on a specific date"""
        try:
            # For now, use current season stats as approximation
            # In a full implementation, this would query historical data up to as_of_date
            season_stats = self.get_player_season_stats(player_id)
            
            if season_stats.empty:
                return {'avg': 0, 'hits': 0, 'plate_appearances': 0}
            
            # Calculate average using plate appearances
            hits = season_stats['hits'].iloc[0] if 'hits' in season_stats.columns else 0
            pa = season_stats['plateAppearances'].iloc[0] if 'plateAppearances' in season_stats.columns else 1
            avg = hits / pa if pa > 0 else 0
            
            return {
                'avg': avg,
                'hits': hits,
                'plate_appearances': pa
            }
        except Exception as e:
            logger.debug(f"Error getting historical season stats for player {player_id}: {e}")
            return {'avg': 0, 'hits': 0, 'plate_appearances': 0}

    def _get_historical_recent_performance(self, player_id, as_of_date, days_back=30):
        """Get recent performance as it would have been on a specific date"""
        try:
            # Calculate the 30-day window ending on as_of_date
            end_date = as_of_date
            start_date = end_date - timedelta(days=days_back)
            
            # For now, use a simplified approach - in production you'd want cached historical data
            recent_perf = self.get_recent_performance(player_id, days_back)
            
            # If we have valid recent performance data, use it
            # Otherwise provide reasonable defaults
            if recent_perf['plate_appearances'] > 0:
                return recent_perf
            else:
                # Fall back to season stats if no recent data
                season_stats = self._get_historical_season_stats(player_id, as_of_date)
                return {
                    'recent_avg': season_stats['avg'],
                    'hits': 0,
                    'plate_appearances': 0
                }
        except Exception as e:
            logger.debug(f"Error getting historical recent performance for player {player_id}: {e}")
            return {'recent_avg': 0, 'hits': 0, 'plate_appearances': 0}

    def _calculate_historical_prediction(self, player_id, pitcher_id, ballpark, season_stats, recent_stats, game_date):
        """Calculate what the prediction would have been on the historical game date"""
        try:
            # Use historical stats to reconstruct the prediction
            season_avg = season_stats.get('avg', 0)
            recent_avg = recent_stats.get('recent_avg', 0)
            
            # Base probability using the same weights as current system
            base_probability = (
                season_avg * self.prediction_weights['season_avg'] +
                recent_avg * self.prediction_weights['recent_performance']
            )
            
            # Apply ballpark factor if available
            if ballpark and ballpark in self.ballpark_factors:
                ballpark_factor = self.ballpark_factors[ballpark]
                base_probability *= ballpark_factor
            
            # For historical predictions, we'll skip weather and detailed matchup analysis
            # since that data is harder to reconstruct accurately
            
            return round(base_probability, 3)
            
        except Exception as e:
            logger.debug(f"Error calculating historical prediction for player {player_id}: {e}")
            return 0.250  # Default fallback

    def get_game_box_score(self, game_pk):
        """Get box score for a game using the correct API endpoint structure"""
        try:
            url = f"{self.mlb.BASE_URL}/game/{game_pk}/boxscore"
            
            response = self.mlb.session.get(url)
            if response.status_code == 404:
                logger.debug(f"Game {game_pk} not found (404 error) - likely future/cancelled game")
                return None
            elif response.status_code != 200:
                logger.error(f"API error for game {game_pk}: {response.status_code}")
                return None
                
            data = response.json()
            
            # The API returns data directly with 'teams' at the root level
            # Structure: {'teams': {'home': {...}, 'away': {...}}}
            if 'teams' not in data:
                logger.warning(f"No teams data in boxscore for game {game_pk}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error getting box score for game {game_pk}: {e}")
            return None

    def log_prediction(self, player_id, player_name, game_date, game_pk, ballpark, 
                      opposing_pitcher_id, season_stats, recent_stats, predicted_prob, 
                      ballpark_factor, weather_factor=1.0, matchup_factor=1.0):
        """Log a prediction in real-time for later training validation"""
        try:
            # Helper function to convert numpy/pandas types to Python types
            def convert_to_python_type(value):
                if hasattr(value, 'item'):  # numpy scalar
                    return value.item()
                elif hasattr(value, 'iloc') and len(value) > 0:  # pandas Series
                    return float(value.iloc[0]) if value.iloc[0] is not None else 0
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    return value
                else:
                    return float(value) if value is not None else 0
            
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'player_id': int(player_id),
                'player_name': str(player_name),
                'game_date': str(game_date),
                'game_pk': int(game_pk) if game_pk else None,
                'ballpark': str(ballpark) if ballpark else None,
                'opposing_pitcher_id': int(opposing_pitcher_id) if opposing_pitcher_id else None,
                'season_avg': convert_to_python_type(season_stats.get('AVG', 0) if isinstance(season_stats, dict) else (season_stats['AVG'].iloc[0] if not season_stats.empty and 'AVG' in season_stats.columns else 0)),
                'season_hits': convert_to_python_type(season_stats.get('hits', 0) if isinstance(season_stats, dict) else (season_stats['hits'].iloc[0] if not season_stats.empty and 'hits' in season_stats.columns else 0)),
                'season_pa': convert_to_python_type(season_stats.get('plateAppearances', 0) if isinstance(season_stats, dict) else (season_stats['plateAppearances'].iloc[0] if not season_stats.empty and 'plateAppearances' in season_stats.columns else 0)),
                'recent_avg': float(recent_stats.get('recent_avg', 0)),
                'recent_hits': int(recent_stats.get('hits', 0)),
                'recent_pa': int(recent_stats.get('plate_appearances', 0)),
                'predicted_prob': float(predicted_prob),
                'ballpark_factor': float(ballpark_factor),
                'weather_factor': float(weather_factor),
                'matchup_factor': float(matchup_factor),
                'prediction_weights': {k: float(v) for k, v in self.prediction_weights.items()}
            }
            
            # Store in Redis with expiration (keep for 30 days)
            cache_key = f"prediction_log:{game_date}:{player_id}:{game_pk}"
            redis_client.setex(cache_key, 60 * 60 * 24 * 30, json.dumps(prediction_record))
            
            # Also store in a daily predictions list for easier retrieval
            daily_key = f"daily_predictions:{game_date}"
            existing_predictions = redis_client.get(daily_key)
            if existing_predictions:
                predictions_list = json.loads(existing_predictions)
            else:
                predictions_list = []
            
            predictions_list.append(prediction_record)
            redis_client.setex(daily_key, 60 * 60 * 24 * 30, json.dumps(predictions_list))
            
            logger.debug(f"Logged prediction for {player_name} ({player_id}): {predicted_prob:.3f}")
            
        except Exception as e:
            logger.error(f"Error logging prediction for player {player_id}: {e}")
            logger.exception("Full traceback:")

    def get_logged_predictions(self, start_date, end_date):
        """Retrieve logged predictions for a date range"""
        try:
            all_predictions = []
            current_date = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            
            while current_date <= end_date_obj:
                date_str = current_date.strftime('%Y-%m-%d')
                daily_key = f"daily_predictions:{date_str}"
                daily_predictions = redis_client.get(daily_key)
                
                if daily_predictions:
                    predictions_list = json.loads(daily_predictions)
                    all_predictions.extend(predictions_list)
                
                current_date += timedelta(days=1)
            
            logger.info(f"Retrieved {len(all_predictions)} logged predictions from {start_date} to {end_date}")
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error retrieving logged predictions: {e}")
            return []

    def merge_predictions_with_outcomes(self, predictions, outcomes_data):
        """Merge logged predictions with actual game outcomes"""
        try:
            merged_data = []
            
            # Create lookup for outcomes by player_id and game_pk
            outcomes_lookup = {}
            for outcome in outcomes_data:
                key = f"{outcome['player_id']}_{outcome['game_pk']}"
                outcomes_lookup[key] = outcome
            
            # Merge predictions with outcomes
            for prediction in predictions:
                lookup_key = f"{prediction['player_id']}_{prediction['game_pk']}"
                
                if lookup_key in outcomes_lookup:
                    outcome = outcomes_lookup[lookup_key]
                    
                    merged_record = prediction.copy()
                    merged_record.update({
                        'actual_hits': outcome.get('actual_hits', 0),
                        'actual_at_bats': outcome.get('actual_at_bats', 0),
                        'actual_pa': outcome.get('actual_pa', 0),
                        'got_hit': outcome.get('got_hit', 0),  # Keep for reference
                        'actual_avg': outcome.get('actual_hits', 0) / max(outcome.get('actual_at_bats', 1), 1),  # Use batting average as target
                        'prediction_error': abs(prediction['predicted_prob'] - (outcome.get('actual_hits', 0) / max(outcome.get('actual_at_bats', 1), 1)))
                    })
                    
                    merged_data.append(merged_record)
            
            logger.info(f"Merged {len(merged_data)} predictions with outcomes out of {len(predictions)} total predictions")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging predictions with outcomes: {e}")
            return []

    def create_train_test_split(self, training_data, test_size=0.2, random_state=42):
        """Create train/test split from training data"""
        try:
            if not training_data or len(training_data) < 10:
                logger.warning("Insufficient data for train/test split")
                return [], [], {}
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(training_data)
            
            # Set random seed for reproducibility
            np.random.seed(random_state)
            
            # Shuffle the data
            df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            
            # Calculate split index
            split_idx = int(len(df_shuffled) * (1 - test_size))
            
            # Split the data
            train_df = df_shuffled[:split_idx]
            test_df = df_shuffled[split_idx:]
            
            # Convert back to list of dictionaries
            train_data = train_df.to_dict('records')
            test_data = test_df.to_dict('records')
            
            # Create split info
            split_info = {
                'total_records': len(training_data),
                'train_records': len(train_data),
                'test_records': len(test_data),
                'test_size': test_size,
                'random_state': random_state,
                'train_hit_rate': train_df['actual_avg'].mean() if 'actual_avg' in train_df.columns else train_df['got_hit'].mean() if 'got_hit' in train_df.columns else 0,
                'test_hit_rate': test_df['actual_avg'].mean() if 'actual_avg' in test_df.columns else test_df['got_hit'].mean() if 'got_hit' in test_df.columns else 0
            }
            
            logger.info(f"Created train/test split: {len(train_data)} train, {len(test_data)} test")
            logger.info(f"Train hit rate: {split_info['train_hit_rate']:.3f}, Test hit rate: {split_info['test_hit_rate']:.3f}")
            
            return train_data, test_data, split_info
            
        except Exception as e:
            logger.error(f"Error creating train/test split: {e}")
            return [], [], {}

    def evaluate_predictions(self, test_data, model_weights=None):
        """Evaluate prediction accuracy on test data"""
        try:
            if not test_data:
                return {}
            
            df = pd.DataFrame(test_data)
            
            # Use provided weights or current weights
            weights = model_weights if model_weights else self.prediction_weights
            
            # If we don't have predicted probabilities, calculate them using the weights
            if 'predicted_prob' not in df.columns:
                logger.info("Calculating predictions for test data using current weights")
                predictions = []
                for _, row in df.iterrows():
                    # Reconstruct prediction using available data
                    season_avg = row.get('season_avg', 0)
                    recent_avg = row.get('recent_avg', 0)
                    ballpark_factor = row.get('ballpark_factor', 1.0)
                    
                    # Calculate prediction using current weights
                    base_prob = (
                        season_avg * weights['season_avg'] +
                        recent_avg * weights['recent_performance']
                    )
                    final_prob = base_prob * ballpark_factor
                    predictions.append(final_prob)
                
                df['predicted_prob'] = predictions
            
            # Calculate evaluation metrics
            actual = df['actual_avg'].values if 'actual_avg' in df.columns else df['got_hit'].values
            predicted = df['predicted_prob'].values
            
            # Basic accuracy metrics
            metrics = {}
            
            # 1. Mean Absolute Error (MAE)
            metrics['mae'] = np.mean(np.abs(predicted - actual))
            
            # 2. Root Mean Square Error (RMSE)
            metrics['rmse'] = np.sqrt(np.mean((predicted - actual) ** 2))
            
            # 3. Correlation coefficient
            if len(set(actual)) > 1 and len(set(predicted)) > 1:  # Need variation for correlation
                correlation = np.corrcoef(predicted, actual)[0, 1]
                metrics['correlation'] = correlation if not np.isnan(correlation) else 0
            else:
                metrics['correlation'] = 0
            
            # 4. Binary classification accuracy (using 0.5 threshold for got_hit if available)
            if 'got_hit' in df.columns:
                binary_actual = df['got_hit'].values
                binary_predictions = (predicted > 0.5).astype(int)
                accuracy = np.mean(binary_predictions == binary_actual)
                metrics['binary_accuracy'] = accuracy
                
                # 5. Precision and Recall for hit predictions
                true_positives = np.sum((binary_predictions == 1) & (binary_actual == 1))
                false_positives = np.sum((binary_predictions == 1) & (binary_actual == 0))
                false_negatives = np.sum((binary_predictions == 0) & (binary_actual == 1))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1_score
            else:
                metrics['binary_accuracy'] = 0
                metrics['precision'] = 0
                metrics['recall'] = 0
                metrics['f1_score'] = 0
            
            # 6. Calibration metrics (how well probabilities match actual rates)
            # Divide predictions into bins and check calibration
            n_bins = 5
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predicted > bin_lower) & (predicted <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = actual[in_bin].mean()
                    avg_confidence_in_bin = predicted[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    total_samples += np.sum(in_bin)
            
            metrics['calibration_error'] = calibration_error
            
            # 7. Baseline comparison (always predict league average)
            league_avg = actual.mean()
            baseline_mae = np.mean(np.abs(league_avg - actual))
            metrics['baseline_mae'] = baseline_mae
            metrics['mae_improvement'] = (baseline_mae - metrics['mae']) / baseline_mae if baseline_mae > 0 else 0
            
            # 8. Additional statistics
            metrics['total_predictions'] = len(test_data)
            metrics['actual_hit_rate'] = actual.mean()
            metrics['predicted_hit_rate'] = predicted.mean()
            metrics['min_prediction'] = predicted.min()
            metrics['max_prediction'] = predicted.max()
            metrics['prediction_std'] = predicted.std()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            logger.exception("Full traceback:")
            return {}

    def cross_validate_model(self, training_data, n_folds=5, random_state=42):
        """Perform k-fold cross-validation to assess model stability"""
        try:
            if not training_data or len(training_data) < n_folds * 2:
                logger.warning("Insufficient data for cross-validation")
                return {}
            
            df = pd.DataFrame(training_data)
            np.random.seed(random_state)
            
            # Shuffle data
            df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            
            # Calculate fold size
            fold_size = len(df_shuffled) // n_folds
            
            cv_results = {
                'fold_metrics': [],
                'mean_metrics': {},
                'std_metrics': {}
            }
            
            logger.info(f"Starting {n_folds}-fold cross-validation with {len(training_data)} samples")
            
            for fold in range(n_folds):
                # Define test indices for this fold
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else len(df_shuffled)
                
                # Create train/test split for this fold
                test_indices = list(range(start_idx, end_idx))
                train_indices = list(range(0, start_idx)) + list(range(end_idx, len(df_shuffled)))
                
                fold_train_data = df_shuffled.iloc[train_indices].to_dict('records')
                fold_test_data = df_shuffled.iloc[test_indices].to_dict('records')
                
                # Train model on this fold
                fold_weights = self.optimize_prediction_weights(fold_train_data)
                
                # Evaluate on test set
                fold_metrics = self.evaluate_predictions(fold_test_data, fold_weights)
                fold_metrics['fold'] = fold + 1
                fold_metrics['train_size'] = len(fold_train_data)
                fold_metrics['test_size'] = len(fold_test_data)
                
                cv_results['fold_metrics'].append(fold_metrics)
                
                logger.info(f"Fold {fold + 1}/{n_folds}: MAE={fold_metrics.get('mae', 0):.4f}, "
                           f"Correlation={fold_metrics.get('correlation', 0):.4f}, "
                           f"Binary Accuracy={fold_metrics.get('binary_accuracy', 0):.4f}")
            
            # Calculate mean and std across folds
            if cv_results['fold_metrics']:
                metric_names = [k for k in cv_results['fold_metrics'][0].keys() 
                               if k not in ['fold', 'train_size', 'test_size'] and 
                               isinstance(cv_results['fold_metrics'][0][k], (int, float))]
                
                for metric in metric_names:
                    values = [fold[metric] for fold in cv_results['fold_metrics'] if metric in fold]
                    if values:
                        cv_results['mean_metrics'][metric] = np.mean(values)
                        cv_results['std_metrics'][metric] = np.std(values)
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            logger.exception("Full traceback:")
            return {}

    def run_model_evaluation(self, days_back=21, test_size=0.2, cv_folds=5, random_state=42):
        """Complete model evaluation pipeline"""
        try:
            logger.info(f"Starting comprehensive model evaluation")
            logger.info(f"Collecting training data from last {days_back} days...")
            
            # Collect training data
            training_data = self.collect_training_data(days_back=days_back)
            
            if not training_data:
                logger.error("No training data available for evaluation")
                return {}
            
            # Validate training data
            is_valid, validation_message = self.validate_training_data(training_data)
            logger.info(f"Training data validation: {validation_message}")
            
            if not is_valid:
                logger.warning("Training data quality issues detected, but proceeding with evaluation")
            
            # Create train/test split
            train_data, test_data, split_info = self.create_train_test_split(
                training_data, test_size=test_size, random_state=random_state
            )
            
            if not train_data or not test_data:
                logger.error("Failed to create train/test split")
                return {}
            
            # Train model on training set
            logger.info("Training model on training set...")
            trained_weights = self.optimize_prediction_weights(train_data)
            
            # Evaluate on test set
            logger.info("Evaluating model on test set...")
            test_metrics = self.evaluate_predictions(test_data, trained_weights)
            
            # Perform cross-validation
            logger.info("Performing cross-validation...")
            cv_results = self.cross_validate_model(training_data, n_folds=cv_folds, random_state=random_state)
            
            # Compare with baseline (current weights)
            logger.info("Comparing with baseline model...")
            baseline_metrics = self.evaluate_predictions(test_data, self.prediction_weights)
            
            # Compile results
            evaluation_results = {
                'dataset_info': {
                    'total_samples': len(training_data),
                    'training_samples': len(train_data),
                    'test_samples': len(test_data),
                    'test_size': test_size,
                    'days_back': days_back
                },
                'split_info': split_info,
                'trained_weights': trained_weights,
                'baseline_weights': self.prediction_weights.copy(),
                'test_metrics': test_metrics,
                'baseline_metrics': baseline_metrics,
                'cross_validation': cv_results,
                'model_comparison': {}
            }
            
            # Compare trained vs baseline model
            if test_metrics and baseline_metrics:
                comparison = {}
                for metric in ['mae', 'rmse', 'correlation', 'binary_accuracy', 'f1_score']:
                    if metric in test_metrics and metric in baseline_metrics:
                        trained_val = test_metrics[metric]
                        baseline_val = baseline_metrics[metric]
                        
                        if metric in ['mae', 'rmse']:  # Lower is better
                            improvement = (baseline_val - trained_val) / baseline_val if baseline_val > 0 else 0
                        else:  # Higher is better
                            improvement = (trained_val - baseline_val) / baseline_val if baseline_val > 0 else 0
                        
                        comparison[metric] = {
                            'trained': trained_val,
                            'baseline': baseline_val,
                            'improvement': improvement,
                            'improvement_pct': improvement * 100
                        }
                
                evaluation_results['model_comparison'] = comparison
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            logger.exception("Full traceback:")
            return {}

    def print_evaluation_results(self, results):
        """Print comprehensive evaluation results in a readable format"""
        try:
            if not results:
                print("No evaluation results to display")
                return
            
            print("\n" + "="*80)
            print("🏟️  MLB HIT TRACKER - MODEL EVALUATION RESULTS")
            print("="*80)
            
            # Dataset Information
            if 'dataset_info' in results:
                info = results['dataset_info']
                print(f"\n📊 DATASET INFORMATION:")
                print(f"   Total samples: {info.get('total_samples', 0)}")
                print(f"   Training samples: {info.get('training_samples', 0)}")
                print(f"   Test samples: {info.get('test_samples', 0)}")
                print(f"   Test size: {info.get('test_size', 0):.1%}")
                print(f"   Data period: Last {info.get('days_back', 0)} days")
            
            # Model Performance Comparison
            if 'model_comparison' in results and results['model_comparison']:
                print(f"\n🎯 MODEL PERFORMANCE COMPARISON:")
                print(f"{'Metric':<20} {'Trained':<12} {'Baseline':<12} {'Improvement':<12}")
                print("-" * 56)
                
                comparison = results['model_comparison']
                for metric, values in comparison.items():
                    metric_name = metric.upper().replace('_', ' ')
                    trained = values['trained']
                    baseline = values['baseline']
                    improvement = values['improvement_pct']
                    
                    # Format numbers appropriately
                    if metric in ['mae', 'rmse']:
                        trained_str = f"{trained:.4f}"
                        baseline_str = f"{baseline:.4f}"
                    elif metric == 'correlation':
                        trained_str = f"{trained:.3f}"
                        baseline_str = f"{baseline:.3f}"
                    else:
                        trained_str = f"{trained:.3f}"
                        baseline_str = f"{baseline:.3f}"
                    
                    improvement_str = f"{improvement:+.1f}%"
                    if improvement > 0:
                        improvement_str = f"✅ {improvement_str}"
                    elif improvement < 0:
                        improvement_str = f"❌ {improvement_str}"
                    else:
                        improvement_str = f"➖ {improvement_str}"
                    
                    print(f"{metric_name:<20} {trained_str:<12} {baseline_str:<12} {improvement_str}")
            
            # Test Set Results
            if 'test_metrics' in results and results['test_metrics']:
                metrics = results['test_metrics']
                print(f"\n📈 TEST SET PERFORMANCE:")
                print(f"   Mean Absolute Error: {metrics.get('mae', 0):.4f}")
                print(f"   Root Mean Square Error: {metrics.get('rmse', 0):.4f}")
                print(f"   Correlation: {metrics.get('correlation', 0):.3f}")
                print(f"   Binary Accuracy: {metrics.get('binary_accuracy', 0):.3f}")
                print(f"   Precision: {metrics.get('precision', 0):.3f}")
                print(f"   Recall: {metrics.get('recall', 0):.3f}")
                print(f"   F1-Score: {metrics.get('f1_score', 0):.3f}")
                print(f"   Calibration Error: {metrics.get('calibration_error', 0):.4f}")
                print(f"   Improvement over baseline: {metrics.get('mae_improvement', 0):.1%}")
            
            # Cross-Validation Results
            if 'cross_validation' in results and results['cross_validation'].get('mean_metrics'):
                cv = results['cross_validation']
                print(f"\n🔄 CROSS-VALIDATION RESULTS (5-Fold):")
                mean_metrics = cv['mean_metrics']
                std_metrics = cv['std_metrics']
                
                print(f"   MAE: {mean_metrics.get('mae', 0):.4f} ± {std_metrics.get('mae', 0):.4f}")
                print(f"   RMSE: {mean_metrics.get('rmse', 0):.4f} ± {std_metrics.get('rmse', 0):.4f}")
                print(f"   Correlation: {mean_metrics.get('correlation', 0):.3f} ± {std_metrics.get('correlation', 0):.3f}")
                print(f"   Binary Accuracy: {mean_metrics.get('binary_accuracy', 0):.3f} ± {std_metrics.get('binary_accuracy', 0):.3f}")
                print(f"   F1-Score: {mean_metrics.get('f1_score', 0):.3f} ± {std_metrics.get('f1_score', 0):.3f}")
            
            # Model Weights Comparison
            if 'trained_weights' in results and 'baseline_weights' in results:
                print(f"\n⚖️  MODEL WEIGHTS COMPARISON:")
                print(f"{'Weight':<20} {'Trained':<12} {'Baseline':<12} {'Change':<12}")
                print("-" * 56)
                
                trained_weights = results['trained_weights']
                baseline_weights = results['baseline_weights']
                
                for weight_name in trained_weights.keys():
                    trained_val = trained_weights[weight_name]
                    baseline_val = baseline_weights.get(weight_name, 0)
                    change = trained_val - baseline_val
                    
                    change_str = f"{change:+.3f}"
                    if abs(change) > 0.05:
                        change_str = f"🔄 {change_str}"
                    elif abs(change) > 0.01:
                        change_str = f"📊 {change_str}"
                    else:
                        change_str = f"➖ {change_str}"
                    
                    print(f"{weight_name.replace('_', ' ').title():<20} {trained_val:.3f} {' ':<8} {baseline_val:.3f} {' ':<8} {change_str}")
            
            # Summary and Recommendations
            print(f"\n💡 SUMMARY & RECOMMENDATIONS:")
            
            if 'model_comparison' in results and results['model_comparison']:
                mae_improvement = results['model_comparison'].get('mae', {}).get('improvement_pct', 0)
                correlation_improvement = results['model_comparison'].get('correlation', {}).get('improvement_pct', 0)
                
                if mae_improvement > 5:
                    print("   ✅ Significant improvement in prediction accuracy!")
                    print("   🔧 Consider adopting the new trained weights.")
                elif mae_improvement > 0:
                    print("   📈 Modest improvement in prediction accuracy.")
                    print("   📊 Monitor performance over more games before adopting.")
                else:
                    print("   ⚠️  Current weights perform better than trained weights.")
                    print("   🔍 Consider collecting more training data or feature engineering.")
                
                if correlation_improvement > 10:
                    print("   🎯 Strong improvement in prediction correlation!")
                elif correlation_improvement < -10:
                    print("   ⚠️  Correlation decreased - model may be overfitting.")
            
            # Data Quality Assessment
            if 'cross_validation' in results and results['cross_validation'].get('std_metrics'):
                std_metrics = results['cross_validation']['std_metrics']
                mae_std = std_metrics.get('mae', 0)
                if mae_std > 0.05:
                    print("   📊 High variance across folds - consider more stable features.")
                else:
                    print("   ✅ Model shows consistent performance across folds.")
            
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Error printing evaluation results: {e}")
            print(f"Error displaying results: {e}")

    def get_ballpark_weather_profile(self, ballpark_name):
        """Get ballpark-specific weather sensitivity profile"""
        try:
            # Ballpark altitude database (from wind direction feature)
            ballpark_altitudes = {
                'Fenway Park': 20, 'Yankee Stadium': 55, 'Oriole Park at Camden Yards': 54,
                'Rogers Centre': 348, 'Tropicana Field': 15, 'Guaranteed Rate Field': 595,
                'Progressive Field': 660, 'Comerica Park': 585, 'Kauffman Stadium': 750,
                'American Family Field': 635, 'Angel Stadium': 160, 'Daikin Park': 65,
                'Oakland Coliseum': 6, 'T-Mobile Park': 134, 'Globe Life Field': 551,
                'Truist Park': 1050, 'Citi Field': 20, 'Citizens Bank Park': 65,
                'Nationals Park': 56, 'LoanDepot Park': 8, 'Wrigley Field': 595,
                'Great American Ball Park': 550, 'PNC Park': 730, 'Busch Stadium': 465,
                'Chase Field': 1090, 'Coors Field': 5280, 'Dodger Stadium': 340,
                'Petco Park': 62, 'Oracle Park': 12, 'Target Field': 815
            }
            
            # Indoor/retractable roof stadiums
            indoor_stadiums = {
                'Tropicana Field', 'Rogers Centre', 'Chase Field', 'LoanDepot Park',
                'Daikin Park', 'American Family Field', 'T-Mobile Park', 'Globe Life Field'
            }
            
            # Coastal stadiums (different weather patterns)
            coastal_stadiums = {
                'Fenway Park', 'Yankee Stadium', 'Oriole Park at Camden Yards', 
                'Citi Field', 'Oracle Park', 'Petco Park', 'LoanDepot Park',
                'T-Mobile Park', 'Tropicana Field'
            }
            
            altitude = ballpark_altitudes.get(ballpark_name, 500)
            is_indoor = ballpark_name in indoor_stadiums
            is_coastal = ballpark_name in coastal_stadiums
            
            # Calculate weather sensitivity factors
            profile = {
                'altitude': altitude,
                'is_indoor': is_indoor,
                'is_coastal': is_coastal,
                'temperature_sensitivity': 1.0,
                'wind_sensitivity': 1.0,
                'precipitation_sensitivity': 1.0,
                'description': []
            }
            
            # Altitude effects
            if altitude >= 5000:  # Mile-high (Coors Field)
                profile['temperature_sensitivity'] = 1.3  # More sensitive to temperature
                profile['wind_sensitivity'] = 1.4  # Much more sensitive to wind
                profile['description'].append(f"Extreme altitude ({altitude}ft) amplifies all weather effects")
            elif altitude >= 3000:  # High altitude
                profile['temperature_sensitivity'] = 1.15
                profile['wind_sensitivity'] = 1.2
                profile['description'].append(f"High altitude ({altitude}ft) enhances weather effects")
            elif altitude >= 1000:  # Moderate altitude
                profile['temperature_sensitivity'] = 1.05
                profile['wind_sensitivity'] = 1.1
                profile['description'].append(f"Moderate altitude ({altitude}ft) slightly enhances weather effects")
            
            # Indoor stadium effects
            if is_indoor:
                profile['temperature_sensitivity'] = 0.1  # Minimal temperature effects
                profile['wind_sensitivity'] = 0.1  # Minimal wind effects
                profile['precipitation_sensitivity'] = 0.0  # No precipitation effects
                profile['description'].append("Indoor/retractable roof stadium minimizes weather impact")
            
            # Coastal effects
            if is_coastal and not is_indoor:
                profile['wind_sensitivity'] *= 1.1  # Coastal winds more variable
                profile['description'].append("Coastal location increases wind variability")
            
            # Special cases
            if ballpark_name == 'Wrigley Field':
                profile['wind_sensitivity'] *= 1.2  # Famous for wind effects from Lake Michigan
                profile['description'].append("Lake Michigan creates unique wind patterns")
            elif ballpark_name == 'Oracle Park':
                profile['wind_sensitivity'] *= 1.3  # Bay winds are notorious
                profile['description'].append("San Francisco Bay winds significantly affect play")
            elif ballpark_name == 'Coors Field':
                profile['temperature_sensitivity'] *= 1.2  # Thin air + temperature
                profile['description'].append("Thin air at altitude makes weather effects more pronounced")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting ballpark weather profile for {ballpark_name}: {e}")
            return {
                'altitude': 500,
                'is_indoor': False,
                'is_coastal': False,
                'temperature_sensitivity': 1.0,
                'wind_sensitivity': 1.0,
                'precipitation_sensitivity': 1.0,
                'description': ['Default weather sensitivity profile']
            }

    def get_direct_matchup_data(self, batter_id, pitcher_id):
        """Get direct matchup statistics - now disabled, only using Retrosheet data"""
        logger.info(f"get_direct_matchup_data called for {batter_id} vs {pitcher_id} - MLB Stats API disabled, using Retrosheet only")
        
        # This method now returns empty data since we only use Retrosheet data
        # The actual matchup logic is handled in get_historical_matchup which uses Retrosheet
        logger.info(f"No MLB Stats API direct matchup lookup for {batter_id} vs {pitcher_id} - using Retrosheet only")
        return pd.DataFrame()

    def is_regular_season_game(self, game):
        """Check if a game is a regular season game (not Spring Training, All-Star, etc.)"""
        game_type = game.get('gameType', '')
        return game_type == 'R'  # 'R' = Regular season, 'S' = Spring Training, 'A' = All-Star, etc.
        
    def get_season_start_date(self, season=None):
        """Get the actual regular season start date, excluding Spring Training"""
        if season is None:
            season = self.current_season
            
        try:
            # Regular season typically starts in late March/early April
            # Look for first regular season games (not Spring Training)
            regular_season_start = datetime(season, 3, 15)  # Conservative estimate
            
            # Try to get more precise date from schedule API
            # Look for first regular season games in March/April
            for day in range(15, 32):  # Check March 15-31
                try:
                    test_date = datetime(season, 3, day).strftime('%Y-%m-%d')
                    games = self.mlb.get_schedule_by_date(test_date)
                    if games:
                        # Check if these are regular season games (not Spring Training)
                        for game in games:
                            game_type = game.get('gameType', '')
                            if game_type == 'R':  # 'R' = Regular season, 'S' = Spring Training
                                actual_start = datetime(season, 3, day)
                                logger.info(f"Found actual regular season start date: {actual_start.strftime('%Y-%m-%d')}")
                                return actual_start
                except Exception as e:
                    logger.debug(f"Error checking date {test_date}: {e}")
                    continue
                    
            # If no regular season games found in March, check April
            for day in range(1, 31):
                try:
                    test_date = datetime(season, 4, day).strftime('%Y-%m-%d')
                    games = self.mlb.get_schedule_by_date(test_date)
                    if games:
                        # Check if these are regular season games
                        for game in games:
                            game_type = game.get('gameType', '')
                            if game_type == 'R':  # Regular season
                                actual_start = datetime(season, 4, day)
                                logger.info(f"Found actual regular season start date: {actual_start.strftime('%Y-%m-%d')}")
                                return actual_start
                except Exception as e:
                    logger.debug(f"Error checking date {test_date}: {e}")
                    continue
                    
            # Fall back to late March estimate for regular season
            logger.warning(f"Could not determine exact regular season start date, using March 25 estimate")
            return datetime(season, 3, 25)
            
        except Exception as e:
            logger.warning(f"Error determining regular season start date: {e}, falling back to March 25")
            return datetime(season, 3, 25)

    async def get_historical_weather(self, lat, lon, date):
        """Get real historical weather data using free APIs"""
        try:
            # Try to get current weather conditions from NWS API (free, no key required)
            nws_data = await self._get_nws_current_weather(lat, lon)
            
            if nws_data:
                # Adjust current conditions for historical context
                return self._adjust_weather_for_historical_date(nws_data, date)
            
            # Fallback to enhanced simulation
            return self._estimate_historical_weather(lat, lon, date)
            
        except Exception as e:
            logger.warning(f"Historical weather lookup failed: {e}")
            return self._estimate_historical_weather(lat, lon, date)
    
    async def _get_nws_current_weather(self, lat, lon):
        """Get current weather from National Weather Service API (free)"""
        try:
            import aiohttp
            headers = {
                'User-Agent': 'MLB-Hit-Tracker/1.0 (your-email@example.com)'  # Required by NWS API
            }
            
            async with aiohttp.ClientSession() as session:
                # First, get the grid point for the coordinates
                points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
                
                async with session.get(points_url, headers=headers) as response:
                    if response.status == 200:
                        points_data = await response.json()
                        
                        # Get the forecast office and grid coordinates
                        forecast_url = points_data['properties']['forecast']
                        
                        # Get current forecast
                        async with session.get(forecast_url, headers=headers) as forecast_response:
                            if forecast_response.status == 200:
                                forecast_data = await forecast_response.json()
                                
                                # Extract current period
                                current_period = forecast_data['properties']['periods'][0]
                                
                                return {
                                    'temperature': current_period['temperature'],
                                    'humidity': current_period.get('relativeHumidity', {}).get('value', 50),
                                    'wind_speed': self._parse_wind_speed(current_period.get('windSpeed', '5 mph')),
                                    'conditions': current_period['shortForecast'],
                                    'source': 'nws_api'
                                }
            
            return None
            
        except Exception as e:
            logger.debug(f"NWS API error: {e}")
            return None
    
    def _parse_wind_speed(self, wind_str):
        """Parse wind speed from NWS format like '5 to 10 mph' or '15 mph'"""
        try:
            import re
            # Extract first number from wind speed string
            numbers = re.findall(r'\d+', wind_str)
            return int(numbers[0]) if numbers else 5
        except:
            return 5
    
    def _adjust_weather_for_historical_date(self, current_weather, historical_date):
        """Adjust current weather conditions to historical averages for the date"""
        try:
            # Calculate seasonal adjustment based on historical date
            month = historical_date.month
            
            # Use climate averages instead of random simulation
            temp = current_weather['temperature']
            
            # Apply simple seasonal adjustment - no random variation
            if month in [12, 1, 2]:  # Winter - slightly cooler
                adjusted_temp = max(10, temp - 5)
            elif month in [6, 7, 8]:  # Summer - use current temp
                adjusted_temp = temp
            else:  # Spring/Fall - moderate adjustment
                adjusted_temp = temp - 2
            
            # Use consistent humidity and wind based on season
            if month in [6, 7, 8]:  # Summer
                humidity = 70
                wind_speed = 8
            elif month in [12, 1, 2]:  # Winter
                humidity = 60
                wind_speed = 12
            else:  # Spring/Fall
                humidity = 65
                wind_speed = 10
            
            logger.info(f"Adjusted weather for {historical_date.strftime('%Y-%m-%d')}: {adjusted_temp:.1f}°F (base: {temp}°F)")
            
            return {
                'temperature': adjusted_temp,
                'humidity': humidity,
                'pressure': 29.92,  # Standard pressure
                'wind_speed': wind_speed,
                'precipitation': 'rain' in current_weather.get('conditions', '').lower(),
                'conditions': current_weather.get('conditions', 'Unknown'),
                'source': 'nws_seasonal_average'
            }
            
        except Exception as e:
            logger.warning(f"Weather adjustment failed: {e}")
            return current_weather
    
    def _estimate_historical_weather(self, lat, lon, date):
        """Get historical weather averages based on location and time of year"""
        try:
            month = date.month
            day_of_year = date.timetuple().tm_yday
            
            # Climate-based temperature averages by latitude and season
            # Using NOAA climate normals as reference
            if lat >= 45:  # Northern regions (e.g., Minneapolis, Seattle)
                if 3 <= month <= 5:  # Spring
                    base_temp = 50
                elif 6 <= month <= 8:  # Summer
                    base_temp = 75
                elif 9 <= month <= 11:  # Fall
                    base_temp = 55
                else:  # Winter
                    base_temp = 25
            elif lat >= 35:  # Mid-latitude regions (e.g., Chicago, New York)
                if 3 <= month <= 5:  # Spring
                    base_temp = 60
                elif 6 <= month <= 8:  # Summer
                    base_temp = 80
                elif 9 <= month <= 11:  # Fall
                    base_temp = 65
                else:  # Winter
                    base_temp = 40
            elif lat >= 25:  # Southern regions (e.g., Atlanta, Phoenix)
                if 3 <= month <= 5:  # Spring
                    base_temp = 70
                elif 6 <= month <= 8:  # Summer
                    base_temp = 85
                elif 9 <= month <= 11:  # Fall
                    base_temp = 75
                else:  # Winter
                    base_temp = 55
            else:  # Subtropical/tropical (e.g., Miami, Tampa)
                if 3 <= month <= 5:  # Spring
                    base_temp = 75
                elif 6 <= month <= 8:  # Summer
                    base_temp = 85
                elif 9 <= month <= 11:  # Fall
                    base_temp = 80
                else:  # Winter
                    base_temp = 70
            
            # Longitude adjustments for continental effects
            if lon < -100:  # Western regions (drier, more temperature variation)
                if month in [6, 7, 8]:  # Summer - hotter in west
                    base_temp += 5
                elif month in [12, 1, 2]:  # Winter - colder in continental west
                    base_temp -= 5
            elif lon > -80:  # Eastern coastal regions (maritime influence)
                if month in [6, 7, 8]:  # Summer - cooler near coast
                    base_temp -= 3
                elif month in [12, 1, 2]:  # Winter - warmer near coast
                    base_temp += 3
            
            # Climate-based humidity averages
            if lat >= 40:  # Northern regions
                humidity = 65 if month in [4, 5, 6, 10, 11] else 60
            elif lon > -90:  # Eastern regions (higher humidity)
                humidity = 75 if month in [6, 7, 8] else 65
            else:  # Western regions (lower humidity)
                humidity = 45 if month in [6, 7, 8] else 50
            
            # Standard atmospheric pressure (varies slightly with altitude)
            # Estimate altitude effect: -0.01 inches per 300 feet elevation
            estimated_elevation = max(0, (lat - 25) * 100)  # Rough elevation estimate
            pressure = 29.92 - (estimated_elevation / 300 * 0.01)
            
            # Seasonal wind speed averages
            if month in [3, 4, 11, 12]:  # Windier months
                wind_speed = 12
            elif month in [6, 7, 8]:  # Calmer summer months
                wind_speed = 8
            else:
                wind_speed = 10
            
            # Seasonal precipitation likelihood (not random, based on climate)
            if lat >= 40:  # Northern regions
                precip_likely = month in [4, 5, 10, 11]  # Spring/fall precipitation
            else:  # Southern regions
                precip_likely = month in [6, 7, 8, 9] if lon > -100 else month in [12, 1, 2]  # Summer thunderstorms (east) or winter rains (west)
            
            logger.info(f"Climate-based weather average for {date.strftime('%Y-%m-%d')} at ({lat:.2f}, {lon:.2f}): {base_temp}°F, {humidity}% humidity")
            
            return {
                'temperature': base_temp,
                'humidity': humidity,
                'pressure': round(pressure, 2),
                'wind_speed': wind_speed,
                'precipitation': precip_likely,
                'source': 'climate_normal_average'
            }
            
        except Exception as e:
            logger.warning(f"Weather average calculation failed: {e}")
            # Fallback to simple defaults based on time of year
            month = date.month
            temp = 70 if 4 <= month <= 10 else 50  # Warm season vs cool season
            return {
                'temperature': temp,
                'humidity': 60,
                'pressure': 29.92,
                'wind_speed': 10,
                'precipitation': False,
                'source': 'simple_seasonal_default'
            }

async def main(tracker=None, min_recent_pa=15, min_season_pa=0):
    """Main execution function"""
    if tracker is None:
        tracker = MLBHitTracker()
    
    start_time = time.time()
    
    try:
        # Calculate hit probabilities for all players
        probabilities = await tracker.get_todays_hit_probabilities(min_recent_pa=min_recent_pa, min_season_pa=min_season_pa)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Probability calculations completed in {execution_time:.2f} seconds")
        logger.info(f"Number of predictions: {len(probabilities)}")
        
        # Configuration information
        logger.info("\n=== CONFIGURATION ===")
        logger.info(f"Minimum recent plate appearances: {min_recent_pa}")
        if min_season_pa > 0:
            logger.info(f"Minimum season plate appearances: {min_season_pa}")
        logger.info(f"Prediction date: {tracker.prediction_date.strftime('%A, %B %d, %Y')}")
        logger.info(f"Total qualified players: {len(probabilities)}")
        
        if not probabilities:
            logger.warning("No players found meeting minimum PA requirements.")
            return
        
        logger.info("\n=== TOP 10 HIGHEST HIT PROBABILITY PLAYERS ===")
        print("\n=== TOP 10 HIGHEST HIT PROBABILITY PLAYERS ===")
        
        # Sort by probability (descending)
        probabilities.sort(key=lambda x: x['probability'], reverse=True)
        
        # Display top 10 players with detailed analysis
        for i, prob in enumerate(probabilities[:10], 1):
            # Get detailed stats for the player
            season_stats = tracker.get_player_season_stats(prob['player_id'])
            recent_perf = tracker.get_recent_performance(prob['player_id'])
            
            # Calculate contributing factors
            ballpark_factor = tracker.ballpark_factors.get(prob['ballpark'], 1.0)
            weather_impact = prob.get('weather_impact', 1.0)
            matchup_advantage = prob.get('matchup_advantage', 1.0)
            
            # Print detailed analysis
            logger.info(f"\n{i}. {prob['player_name']} ({prob['team']})")
            logger.info(f"   Overall Hit Probability: {prob['probability']:.3f}")
            logger.info(f"   Game: {prob['game_info']} at {prob.get('game_time', 'TBD')}")
            logger.info(f"   Ballpark: {prob['ballpark']} (Factor: {ballpark_factor:.3f})")
            logger.info(f"   Opposing Pitcher: {prob.get('opposing_pitcher', 'TBD')}")
            
            # Show doubleheader advantage if applicable
            doubleheader_advantage = prob.get('doubleheader_advantage', 1.0)
            doubleheader_explanation = prob.get('doubleheader_explanation', '')
            
            if doubleheader_advantage > 1.0:
                logger.info(f"   🎯 DOUBLEHEADER ADVANTAGE: {doubleheader_advantage:.3f}x factor ({doubleheader_explanation})")
            elif 'doubleheader' in doubleheader_explanation.lower() and doubleheader_advantage == 1.0:
                logger.info(f"   ℹ️  Doubleheader Status: {doubleheader_explanation}")
            # If no doubleheader, don't show anything
            
            logger.info("\n   Season Stats:")
            if not season_stats.empty:
                hits = season_stats['hits'].iloc[0]
                pa = season_stats['plateAppearances'].iloc[0]
                # Calculate average on-the-fly from raw stats
                avg = hits / pa if pa > 0 else 0.0
                logger.info(f"    - AVG: {avg:.3f}")
                logger.info(f"    - Hits: {hits}")
                logger.info(f"    - PA: {pa}")
            
            logger.info("\n   Recent Performance (Last 30 days):")
            logger.info(f"    - Hits: {recent_perf['hits']}")
            logger.info(f"    - Plate Appearances: {recent_perf['plate_appearances']}")
            logger.info(f"    - Average: {recent_perf['recent_avg']:.3f}")
            logger.info("")
        
        # Summary statistics
        avg_probability = sum(p['probability'] for p in probabilities) / len(probabilities)
        max_probability = probabilities[0]['probability']
        min_probability = probabilities[-1]['probability']
        
        logger.info(f"\n=== SUMMARY STATISTICS ===")
        logger.info(f"Average Hit Probability: {avg_probability:.3f}")
        logger.info(f"Highest Probability: {max_probability:.3f} ({probabilities[0]['player_name']})")
        logger.info(f"Lowest Probability: {min_probability:.3f} ({probabilities[-1]['player_name']})")
        
        logger.info(f"\nTotal execution time: {execution_time:.2f} seconds")
        
    finally:
        # Clean up async resources
        await tracker.cleanup()

def configure_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('performance.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MLBHitTracker')

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLB Hit Probability Tracker')
    parser.add_argument('prediction_date', nargs='?', default=None, 
                        help='Prediction date in YYYY-MM-DD format (default: tomorrow)')
    parser.add_argument('--min-pa', type=int, default=15,
                        help='Minimum plate appearances in last 30 days (default: 15)')
    parser.add_argument('--min-season-pa', type=int, default=0,
                        help='Minimum season plate appearances (default: 0)')
    
    args = parser.parse_args()
    
    # Initialize the tracker
    if args.prediction_date:
        tracker = MLBHitTracker(prediction_date=args.prediction_date)
    else:
        tracker = MLBHitTracker()
    
    # Run the tracker with specified minimum PA requirements
    asyncio.run(main(tracker, min_recent_pa=args.min_pa, min_season_pa=args.min_season_pa))