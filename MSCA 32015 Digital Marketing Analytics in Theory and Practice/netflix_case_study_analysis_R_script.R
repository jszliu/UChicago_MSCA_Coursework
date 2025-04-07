#Netflix Case Study R Script for Analysis of IMDb Data
#always useful R tips & tricks found here: https://impaulchung.wordpress.com/2013/01/09/r-studio-pauls-tips-tricks/

# This analysis is based on the case study "‘What's the Next Big Thing?’: Netflix and Analyzing Data for Insights'' written by Kevin Hartman (found here: https://artscience.blog/home/netflix-case-study). The purpose of this script is to consolidate downloaded IMDb data into a tidy dataframes and then conduct simple analysis to help answer the key question: “What emerging consumer interest trends present opportunities for Netflix?”

#install necessary packages
install.packages("tidyverse") #includes "ggplot2","purrr","tibble","dplyr","tidyr","stringr","readr","forcats"
install.packages("gpairs")
install.packages("grid")
install.packages("lattice")
install.packages("arsenal")

#select libraries (i.e., flip them "on")
library(tidyverse)
library(gpairs)
library(grid)
library(lattice)
library(arsenal)

############################
# STEP 1: COLLECT THE DATA #
############################

#get data from IMDb https://datasets.imdbws.com/
#Each dataset is contained in a gzipped, tab-separated-values (TSV) formatted file in the UTF-8 character set. The first line in each file contains headers that describe what is in each column. A ‘\N’ is used to denote that a particular field is missing or null for that title/name, so we need to change that to something that R recognizes as a missing value: 'na'.
#Documentation for these data files can be found on http://www.imdb.com/interfaces/
#Here's info from Datacamp on importing .tsv files: https://campus.datacamp.com/courses/reading-data-into-r-with-readr/importing-data-with-readr?ex=2
imdb_name.basics <- read_tsv("~/Downloads/name.basics.tsv", na = "\\N", quote = '')
imdb_title.episode <- read_tsv("~/Downloads/title.episode.tsv", na = "\\N", quote = '')
imdb_title.ratings <- read_tsv("~/Downloads/title.ratings.tsv", na = "\\N", quote = '')
imdb_title.akas <- read_tsv("~/Downloads/title.akas.tsv", na = "\\N", quote = '')
imdb_title.crew <- read_tsv("~/Downloads/title.crew.tsv", na = "\\N", quote = '')
imdb_title.basics <- read_tsv("~/Downloads/title.basics.tsv", na = "\\N", quote = '')
imdb_title.principals <- read_tsv("~/Downloads/title.principals.tsv", na = "\\N", quote = '')
#Note that after loading the datasets it may take a bit of time before they show up in the Global Environment. The “principals” dataset alone is a 1.28GB file! It took ~5 minutes for my Mac laptop to load and display these datasets in the Global Environment.


#####################################
# STEP 2: WRANGLE AND TIDY THE DATA #
#####################################

#inspect the data you've collected (other functions like dim(), head(), colnames() are good here, too )
str(imdb_name.basics)
str(imdb_title.episode)
str(imdb_title.ratings)
str(imdb_title.akas)
str(imdb_title.crew)
str(imdb_title.basics)
str(imdb_title.principals)

#prepare data to combine into one dataframe
colnames(imdb_title.akas)
imdb_title.akas <- rename(imdb_title.akas, c("tconst" = "titleId"))
colnames(imdb_title.akas)

#limit analysis to titles that were launched in last three years
df_titles <- filter(imdb_title.basics, imdb_title.basics$startYear > 2017) #1,089,264 observations of 9 variables

#exclude the ... ahem ... adult films
df_titles <- filter(df_titles, df_titles$isAdult == 0) #1,053,382 observations of 9 variables

#create a list of the movie universe with which we will be working ... this will come in handy later
df_universe <- select(df_titles, tconst)

#add other attributes to titles dataframe (note there's not much to "akas" and "principals" will come later on)
df_titles <- left_join(df_titles, imdb_title.ratings) #1,053,382 observations of 11 variables
df_titles <- left_join(df_titles, imdb_title.episode) #1,053,382 observations of 14 variables
df_titles <- left_join(df_titles, imdb_title.crew) #1,053,382 observations of 16 variables

#build a 'Q Score' metric which is defined as (averageRating X numVotes)
df_titles <- mutate(df_titles, Qscore = averageRating * numVotes) #1,053,382 observations of 17 variables

#build dataframes for roles on titles (#principales has ~43M rows, so limit it to only films in our universe)
df_principals <- left_join(df_universe, imdb_title.principals) #this is the filtering step
df_actors <- filter(df_principals, str_detect(category, "actor|actress"))
df_directors <- filter(df_principals, str_detect(category, "director"))
df_writers <- filter(df_principals, str_detect(category, "writer"))
df_producers <- filter(df_principals, str_detect(category, "producer"))

#add real names (and additional data) back in for easier recognition
df_actors <- left_join(df_actors, imdb_name.basics)
df_directors <- left_join(df_directors, imdb_name.basics)
df_writers <- left_join(df_writers, imdb_name.basics)
df_producers <- left_join(df_producers, imdb_name.basics)

#build a dataframe of lead actors from the principals dataset for analysis
df_leads <- df_principals %>% #this symbol (%>%) is called a "pipe" and allows a sequence of multiple operations
  filter(str_detect(category, "actor|actress")) %>% #note that 'self' returns non-actors such as news anchors
  left_join(imdb_name.basics) %>% #this brings in the actual name of the actor for easier recognition
  select(tconst, ordering, nconst, category, primaryName) %>% #selecting a few columns to keep the file size down 
  filter(ordering == min(ordering)) #this is what distinguishes leads from other actors on the title

#let's create a dataframe of actors' number of appearances in lead roles
df_leadsAppearances <- df_leads %>%
  group_by(nconst) %>% #adding nconst as a matching key for later
  count(primaryName, sort = TRUE) %>% #this line counts the number of appearances as a lead by actor
  rename(c("numApps" = "n")) #the count will be under a header of 'n', which can be confusing ... change the header to a more unique name like "numApps" (i.e., "Number of Appearances")

#let's create a dataframe of actors' leadQscore when they were in lead roles (plus some other data)
df_leadsQscores <- df_leads %>%
  left_join(df_titles) %>%
  group_by(nconst) %>% #adding nconst as a matching key for later
  summarize(leadQscore = mean(Qscore, na.rm=TRUE)) %>% #summarize allows for different calculations, like mean
  arrange(desc(leadQscore), by_group = FALSE) %>% #sorting by leadQscore in descending order
  left_join(df_actors) %>% #adding back all the additional data found in df_actors
  left_join(df_leadsAppearances) #adding the number of appearances from df_leadsAppearances

#let's create a dataframe of directors' Qscore when they were in lead roles (plus some other data)
df_directorsQscores <- df_directors %>%
  left_join(df_titles) %>%
  group_by(nconst) %>% #adding nconst as a matching key for later
  summarize(dirQscore = mean(Qscore, na.rm=TRUE)) %>% #summarize allows for different calculations, like mean
  arrange(desc(dirQscore), by_group = FALSE) %>% #sorting by dirQscore in descending order
  left_join(df_directors) #adding back all the additional data found in df_directors

#let's create a dataframe of writers' Qscore when they were in lead roles (plus some other data)
df_writersQscores <- df_writers %>%
  left_join(df_titles) %>%
  group_by(nconst) %>% #adding nconst as a matching key for later
  summarize(wriQscore = mean(Qscore, na.rm=TRUE)) %>% #summarize allows for different calculations, like mean
  arrange(desc(wriQscore), by_group = FALSE) %>% #sorting by wriQscore in descending order
  left_join(df_writers) #adding back all the additional data found in df_writers

#let's create a dataframe of producers' Qscore when they were in lead roles (plus some other data)
df_producersQscores <- df_producers %>%
  left_join(df_titles) %>%
  group_by(nconst) %>% #adding nconst as a matching key for later
  summarize(proQscore = mean(Qscore, na.rm=TRUE)) %>% #summarize allows for different calculations, like mean
  arrange(desc(proQscore), by_group = FALSE) %>% #sorting by proQscore in descending order
  left_join(df_producers) #adding back all the additional data found in df_producers

#let's create a nice, small dataframe of top titles and all the quantifiable data we have so far
df_titlesTop <- df_titles %>%
  arrange(desc(Qscore)) %>% #sorting by leadQscore in descending order
  slice(1:100) %>%
  left_join(df_directorsQscores, by = "tconst") %>%
  left_join(df_leadsQscores, by = "tconst") %>%
  left_join(df_producersQscores, by = "tconst") %>%
  left_join(df_writersQscores, by = "tconst") %>%
  select(c(tconst, startYear, endYear, runtimeMinutes, averageRating, numVotes, Qscore, dirQscore, leadQscore, numApps, proQscore, wriQscore)) %>%
  group_by(tconst) %>%
  summarise_all(funs(mean)) 


########################################
# STEP 3: PERFORM DESCRIPTIVE ANALYSIS #
########################################

#let's look at some summary statistics about our titles
summary(df_titles)
#Min: Lowest value
#1st Q: 25th percentile value
#Median: Middle value
#Mean: Average value
#3rd Q: 75th percentile value
#Max: Highest value
#NA's: Empty values

#let's see what some of these outliers are
filter(df_titles, runtimeMinutes == 28643)
filter(df_titles, startYear == 2115)

#let's look at a top 10 list of titles by Qscore
head(arrange(df_titles,desc(Qscore)), n = 10) #takes a minute or two...

#let's look at some summary statistics about actors' appearances
summary(df_leadsAppearances)

#let's see a top ten list of the actors in terms of appearances in our universe
head(arrange(df_leadsAppearances,desc(numApps)), n = 10)

#let's look at some summary statistics about actors' leadQscores
summary(df_leadsQscores) #provides summary statistics on the columns of the dataframe

#let's see a top ten list of the actors in terms of leadQscore in our universe
head(arrange(df_leadsQscores,desc(leadQscore)), n = 10)


########################################
# STEP 4: PERFORM INFERENTIAL ANALYSIS #
########################################

#use the sample( ) function to take a random sample of titles outside our universe as a baseline
#let's take a random sample of size 400 from a dataset mydata (and not replace the data)
df_titlesSample <- imdb_title.basics[sample(1:nrow(imdb_title.basics), 400, replace=FALSE),]

#now let's add in the ratings for these titles and calculate a Qscore
df_titlesSample <- df_titlesSample %>%
  left_join(imdb_title.ratings) %>%
  mutate(Qscore = averageRating * numVotes)

#let's look at some summary statistics for this sample and our universe
summary(df_titlesSample)
summary(df_titles)

#these operations, from the "arsenal" library, can help test for bias in your sample
#learn more here: https://cran.r-project.org/web/packages/arsenal/vignettes/comparedf.html
comparedf(df_titlesSample, df_titles)
#note the "summary" function takes a long time to run, so I'm skipping it  ... just deleted the "#" to run! 
#summary(comparedf(df_titlesSample, df_titles, by = "tconst"))


########################################
# STEP 5: PERFORM EXPLORATORY ANALYSIS #
########################################

#let's use some simple visualizations to see patterns in our data
hist(df_titles$numVotes) #distribution of the number of votes
hist(df_titles$averageRating) #distribution of average rating
plot(df_titles$numVotes, df_titles$averageRating) #plots data as X,Y
qplot(df_titles$numVotes, df_titles$averageRating) #qplot is the basic plotting function in the ggplot2 package
ggplot(df_titles, aes(x=numVotes, y=averageRating)) + geom_point() #this just makes the plot more intuitive

#let's take a broad look at correlations in our data, but first we need to tidy our data a bit more
df_gpairsData <- df_titlesTop %>%
  select(c(runtimeMinutes, averageRating, numVotes, Qscore, dirQscore, leadQscore, numApps, proQscore, wriQscore))

#now convert this dataframe to a data.matrix
df_gpairsData <- data.matrix(df_gpairsData)

#now ... behold the magic of R!
gpairs(df_gpairsData) #okay, maybe not too much is surprising here, but it's really cool all the same!


###################################
# STEP 6: PERFORM CAUSAL ANALYSIS #
###################################

#is there a relationship between an actor's leadQscore and number of appearances?
qplot(df_leadsQscores$leadQscore, df_leadsQscores$numApps) #qplot is the basic plotting function in ggplot2

#let's use a regression model to see if appearances have an effect on Qscore 
#lm([dependent (Y) variable] ~ [independent (X) variables], data = [data source])
lm_QA = lm(leadQscore ~ numApps, data = df_actors_scores)
lm_QA$coefficients #check the relationship ... can good actors be more selective?

#how about the other way around ... Qscore driving appearances?
lm_AQ = lm(numApps ~ leadQscore, data = df_leadsQscores)
lm_AQ$coefficients #check the relationship ... overexposure?


####################################################
# STEP 7: EXPORT SUMMARY FILE FOR FURTHER ANALYSIS #
####################################################

#create a csv file that can be visualized in Excel, Tableau, or presentation software
#N.B.: This file location is for a Mac. If you are working on a PC, change the file location accordingly (most likely "C:\Users\YOUR_USERNAME\Desktop\...") to export the data. You can read more here: https://datatofish.com/export-dataframe-to-csv-in-r/
write.csv(df_titlesTop, file = '~/Desktop/titlesTop.csv') #small file suitable for Excel or Powerpoint
write.csv(df_gpairsData, file = '~/Desktop/gpairsData.csv') #second smaller file for Excel or Powerpoint
write.csv(df_titles, file = '~/Desktop/titles.csv') #larger file for Tableau
write.csv(df_leadsQscores, file = '~/Desktop/leadsQscore.csv') #second larger file for Tableau

#You're done! Congratulations!
