# MLB Counting to 57 Hits

A statistical analysis of whether MLB's "Beat the Streak" contest can be won — and why it almost certainly cannot.

## The Challenge

MLB's Beat the Streak is a free fantasy contest with a $5.6 million prize. The rules are simple: pick one player each day who you believe will record at least one hit. String together 57 correct picks in a row — beating Joe DiMaggio's 56-game hitting streak — and you win.

Since the contest launched in 2001, nobody has won. This project investigates why.

## Approach

We built predictive models to identify the best hitter to pick on any given day, drawing from three data sources:

- **Statcast pitch-by-pitch data (2017-2021)** — plate discipline, contact rates, batted ball quality
- **MLB Stats API** — game logs, matchup history, platoon splits
- **Visual Crossing weather data + venue locations** — temperature, wind, humidity, and park factors (Parks.csv)

We tested several classifiers (Decision Trees, Random Forests, Logistic Regression) trained on per-game features, with the target variable being whether a player recorded at least one hit in a given game. Models were sorted and filtered by plate appearances per game to favor everyday hitters with more opportunities to get a hit.

## Key Results

Our best-performing model — a Decision Tree filtered by plate appearances per game — achieved:

- **14-day simulated streak** (best run)
- **~71.4% daily success rate** across the evaluation period

A 71.4% daily success rate sounds strong. It is not enough.

The probability of hitting 57 correct picks in a row at 71.4% per day:

```
0.714^57 = 0.0000000007  (~0.00000007%)
```

For context, you would need a **daily success rate above 98%** to have even a coin-flip chance of completing a 57-day streak. No model we tested — and likely no model built on public data — comes close to that threshold.

The mathematical structure of the problem, not the quality of the model, is what makes Beat the Streak essentially unwinnable.

## Data Sources

| Source | Description |
|--------|-------------|
| [Statcast (Baseball Savant)](https://baseballsavant.mlb.com/statcast_search) | Pitch-by-pitch data, 2017-2021 |
| [MLB Stats API](https://statsapi.mlb.com) | Game logs, player stats, matchup data |
| [Visual Crossing](https://www.visualcrossing.com/) | Historical weather data by venue and date |
| Parks.csv | Venue names and geographic coordinates |

## Getting Started

The analysis lives in two Jupyter notebooks:

```bash
pip install jupyter pandas scikit-learn numpy
jupyter notebook main.ipynb
```

`main.ipynb` contains the primary modeling pipeline. `notebook.ipynb` contains supporting exploration and data preparation. `Presentation.pdf` summarizes the findings.

### API Key

The weather data collection in `notebook.ipynb` requires a [Visual Crossing](https://www.visualcrossing.com/) API key. Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and replace 'your-key-here' with your actual API key
```

Then set the environment variable before running the notebook:

```bash
export VISUAL_CROSSING_API_KEY=your-key-here
```
