7.24 - I started by deriving game by game data from retrosheets play by play data. This also included weather data which seems important based on the beat the streak subreddit (temp, sun, winddir, windspeed). Unfortunately this dataset doesn't include 2021. After looking around at various options including a baseball-reference scrape, at this point I believe my best option is to use the statcast api (undocumented, smh) through the pybaseball library which seems popular among baseball researchers. At this moment I've saved the pitch by pitch data for 2019 in a csv. I'll likely want to suplmement this data with weather data to make up for the loss of such.

the statcast data does seem to have everything else I'd need to navigate the info properly with the exception of game times and daynight. 

WORKFLOW:

for every row in statcast - grab game_pk - dump into statsapi.schedule - grab starttime, venue - translate venue into lat, long with coord.csv - dump into visual crossing - grab weather info - dump starttime, venue, and weather info back into statcast df - grab pitcher id from df - run through pybaseball.statcast_pitcher to grab "relevent" stats through DAY BEFORE GAME - dump stats in df - grab batter id from df - run through pybaseball.statcast_batter to grab "relevent" stata through DAY BEFORE GAME - dump stats in df

IMPLEMENT CACHING - have relevent parts of code check their caches first:

save game_pk results and their matching starttime, location, weather - no need to ever clear

save pitcher and batter stats - clear after every day (not game because you can't wager between double headers)


according to retrosheets the day/night cutoff is 5PM


stadium coordinates from https://github.com/sertalpbilal/traveling-baseball-fan-problem/blob/master/data/coords.csv