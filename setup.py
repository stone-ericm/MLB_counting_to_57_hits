#!/usr/bin/env python3
"""
Setup and initialization script for MLB Hit Tracker.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up MLB Hit Tracker environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def setup_configuration():
    """Set up configuration files."""
    print("⚙️ Setting up configuration...")
    
    # Create environment-specific config directories if they don't exist
    config_dirs = ["config/dev", "config/staging", "config/prod"]
    for config_dir in config_dirs:
        Path(config_dir).mkdir(parents=True, exist_ok=True)
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# MLB Hit Tracker Environment Configuration
# Copy this file and customize for your environment

# Environment (development, staging, production)
MLB_TRACKER_ENV=development

# Database
MLB_DB_PATH=mlb_stats_dev.db

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Logging
LOG_LEVEL=INFO

# Debug mode
DEBUG=true

# Monitoring (optional)
ALERT_WEBHOOK_URL=

# API Rate Limits (requests per minute)
MLB_API_RATE_LIMIT=100
WEATHER_API_RATE_LIMIT=50
"""
        env_file.write_text(env_content)
        print("✅ Created .env configuration file")
    
    return True

def setup_database():
    """Initialize the database."""
    print("🗄️ Setting up database...")
    
    try:
        # Import after environment is set up
        sys.path.insert(0, str(Path.cwd()))
        from src.database import setup_database
        from src.config import get_config
        
        # Initialize database
        db_manager = setup_database()
        print("✅ Database initialized successfully")
        
        # Get database stats
        stats = db_manager.get_database_stats()
        print(f"📊 Database file size: {stats.get('file_size_mb', 0):.2f} MB")
        print(f"📊 Schema version: {stats.get('schema_version', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def setup_monitoring():
    """Set up monitoring and logging."""
    print("📊 Setting up monitoring...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.monitoring import setup_monitoring
        
        # Initialize monitoring
        monitoring = setup_monitoring()
        print("✅ Monitoring system initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if sys.platform == "win32":
        pytest_cmd = "venv\\Scripts\\pytest"
    else:
        pytest_cmd = "venv/bin/pytest"
    
    test_command = f"{pytest_cmd} tests/ -v --cov=src --cov-report=html --cov-report=term"
    
    if run_command(test_command, "Running test suite"):
        print("✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        return True
    else:
        print("❌ Some tests failed")
        return False

def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("🔧 Setting up pre-commit hooks...")
    
    if sys.platform == "win32":
        precommit_cmd = "venv\\Scripts\\pre-commit"
    else:
        precommit_cmd = "venv/bin/pre-commit"
    
    # Create .pre-commit-config.yaml if it doesn't exist
    precommit_config = Path(".pre-commit-config.yaml")
    if not precommit_config.exists():
        config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-redis]
"""
        precommit_config.write_text(config_content)
        print("✅ Created .pre-commit-config.yaml")
    
    if run_command(f"{precommit_cmd} install", "Installing pre-commit hooks"):
        return True
    else:
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("📝 Creating sample data...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.database import get_db_manager
        
        db_manager = get_db_manager()
        
        # Store sample park factors
        sample_parks = [
            ("Fenway Park", 1.031, 3, 0.95),
            ("Yankee Stadium", 1.014, 147, 0.92),
            ("Coors Field", 1.235, 19, 0.98),
            ("Petco Park", 0.870, 2680, 0.89)
        ]
        
        for park_name, factor, venue_id, confidence in sample_parks:
            db_manager.store_park_factor(park_name, factor, venue_id, 1000, confidence)
        
        print("✅ Sample park factors created")
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def validate_setup():
    """Validate the setup by running basic functionality tests."""
    print("🔍 Validating setup...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        # Test configuration loading
        from src.config import get_config
        config = get_config()
        print(f"✅ Configuration loaded for environment: {config.environment}")
        
        # Test database connection
        from src.database import get_db_manager
        db_manager = get_db_manager()
        stats = db_manager.get_database_stats()
        print(f"✅ Database connection successful")
        
        # Test API clients (without making actual requests)
        from src.api import get_mlb_client, get_weather_client
        mlb_client = get_mlb_client()
        weather_client = get_weather_client()
        print("✅ API clients initialized")
        
        # Test monitoring
        from src.monitoring import get_logger, get_metrics
        logger = get_logger("setup_validation")
        metrics = get_metrics()
        logger.info("Setup validation test")
        metrics.increment("setup.validation.success")
        print("✅ Monitoring system functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup validation failed: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="MLB Hit Tracker Setup")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-precommit", action="store_true", help="Skip pre-commit setup")
    parser.add_argument("--dev-only", action="store_true", help="Development setup only")
    
    args = parser.parse_args()
    
    print("⚾ MLB Hit Tracker Setup")
    print("=" * 50)
    
    success = True
    
    # Core setup steps
    if not setup_environment():
        success = False
    
    if success and not setup_configuration():
        success = False
    
    if success and not setup_database():
        success = False
    
    if success and not setup_monitoring():
        success = False
    
    if success and not create_sample_data():
        success = False
    
    # Development setup steps
    if success and not args.dev_only:
        if not args.skip_precommit and not setup_pre_commit():
            success = False
        
        if success and not args.skip_tests and not run_tests():
            success = False
    
    # Final validation
    if success and not validate_setup():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and customize .env file")
        print("2. Run: python -m src.main --help")
        print("3. Check docs/ directory for documentation")
        print("4. Run tests: pytest tests/")
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 