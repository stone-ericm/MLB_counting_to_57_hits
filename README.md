# Counting to 57

## [Presentation](https://github.com/stonehengee/phase-3-project/blob/main/README.md])

## Goal

In this project I aim to win MLB's [Beat the Streak](https://www.mlb.com/apps/beat-the-streak) fantasy baseball contest. To do that, I have to pick at least one baseball player every day who I think will get at least one hit. If I'm right for 57 days in a row, I win the contest and with it the $5.6 million cash prize. All the work described below can be seen in [notebook.ipynb](https://github.com/stonehengee/phase-3-project/blob/main/notebook.ipynb) and the data for baseball venue locations can be found in [Parks.csv](https://github.com/stonehengee/phase-3-project/blob/main/Parks.csv).

## Data

To accomplish this I gathered data from [Statcast](https://baseballsavant.mlb.com/statcast_search), [Visual Crossing](https://www.visualcrossing.com/) and the MLB Stats API. I utilized [pybaseball](https://github.com/jldbc/pybaseball) and the [MLB-StatsAPI library](https://github.com/toddrob99/MLB-StatsAPI) in addition to traditional data science libraries like numpy and Pandas.

From Statcast, I gathered pitch-by-pitch data from every game between 2017 and June 30th, 2021. To supplement this, I used the MLB-StatsAPI to gather certain pieces of metadata about each game which wasn't already included in Statcast, most notably probable pitchers. To this, I then added weather data from [Visual Crossing](https://www.visualcrossing.com/). Additionally I gathered latitude, longitude, and altitude data from Wikipedia, Google, as well as [traveling-baseball-fan-problem](https://github.com/sertalpbilal/traveling-baseball-fan-problem/blob/master/data/coords.csv).

Lastly I used all of this data to derive certain statistics I thought might be useful for modeling, including:

- For batters
	- Plate appearances per game
	- Hits per plate appearance against right-handed and left-handed pitchers
	- Average launch angle
	- Average launch speed
- For pitchers
	 - Average plate appearances faced per game played
	 - Average hits given up per inning
	 - Hits given up per plate appearance against right-handed and left-handed batters

## Results

Before running any models, I checked how successful it would be to use the metrics I already had to determine the best player pick for each day. My best result came from ordering each day's batters by their plate appearances per game over the last 2 years. This gave me a daily pick success rate of roughly 62.5% with an all time best streak of 8 days.

Overall my best model was a Decision Tree where the predicted hitters were then sorted by plate appearances per game. This gave me a streak of 14 days with a daily success rate of about 71.4%.

## Conclusion

While all this is a good start, this project is nowhere near finished. While working through the data, I had to sacrifice many possible metrics and influences which could possibly increase my chances. Given more time and resources, I'd add in features that address areas such as:

- More in depth batter/pitcher match-ups
- Pitch types, velocity, and movement
- Bullpens (in addition to probable starting pitchers)
- The ability to pick two players on a given day, should a certain odds threshold be met
- Excluding most pitchers as their strengths lie in areas other than hitting (with the exception of two-way players such as Shohei Ohtani)

## Sources

- [Statcast](https://baseballsavant.mlb.com/statcast_search)
- [Visual Crossing](https://www.visualcrossing.com/)
- [MLB-StatsAPI](https://statsapi.mlb.com/api)
- [traveling-baseball-fan-problem](https://github.com/sertalpbilal/traveling-baseball-fan-problem/blob/master/data/coords.csv)
