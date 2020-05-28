# General Assembly Data Science Capstone Project - Book Review Recommender

The purpose of this project was to build a book review recommender system. 

## Background and Problem Statement
Given the overwhelming number of options for each product with the rise of e-commerce, more and more customers are starting to put weight on reviews. Currently Amazon and Goodreads order their reviews by the reviewer ranking or the number of likes/helpful votes a post gets. But shouldn't how similar a reviewer is to you also have a part to play? In the age of recommmender systems, it felt fitting to try to not build a system that recommends the product as well as recommending the best reviews to read, based on the user similarities. I wanted to measure what sort of impact this re-ordering might have. 

I chose books as this is a problem I have often thought about when reading book reviews and is more applicable to books since with other products you may just be concerned more with the functionality of the product, but with a book you are looking for an experience which is a lot more personal and hence reviews ordered by user-similarity should be of more relevance. 

## Data Accquring 
Decided to use the [dataset provided by Julian McAuley and his team at USCD.](http://jmcauley.ucsd.edu/data/amazon/) I used the 2014 5-core book review dataset as the later 2018 ones were a lot larger. I also downloaded the book metadata dataset. One thing to note here is that when downloading the file, make sure to discard any rows of data that have a title longer than 100 characters as some of the rows contain html code scraped erroneously and causes the last 100k rows to amount to over 5GB when unzipping. I have included the notebook to help you deal with this issue or you can just use the CSV files I used for ease. 

### Data files:
- subset of the data 
- subset of the meta data. 

### Files included
- Data cleaning: incase you want to download the files and have a go at cleaning yourself and storing them as CSVs 
- Data condensing: I decided to only use a subset of the data given time constraints and laptop capabilities. I condensed the dataset down to picking the books that had atleast 50 reviews and then of that choosing the reviewers that had reviewed book atleast 50 times. Left me with ~270k rows of reviews and ratings. Includes the stats of the subset. 
- EDA: Some EDA. Shows that negative reviews get more attention and other findings. 
- Recommender system and Surprise CV test: The notebook for first checking all the algorithms and then gridsearching the chosen 4. 
- Cosine similarities function script: A script containing a random generator function that allow to pick a user and randomly one of their top 10 books. Function returns the current order of book, recommended order, comparision of categories. 
- Results and metrics: metrics to measure the impact of the model as well as visualising the result of the re-ordering using the cosine similarities function 


## Results and next steps 
I obviously cannot measure the impact of the model as I would need customers to tell me if this reordering is helpful or not but it was interesting to see if the model had any massive impact. 

- Group the categories a bit better
- Use better meta dataset 
- try to include review text context analysis to improve the recommender system 
- neighbourhood models to use for cosine similarity rather than calculating using the whole user-item matrix 


# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 

