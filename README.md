# General Assembly Data Science Capstone Project - Book Review Recommender

The purpose of this project was to build a book review recommender system, a system that recommends books to users and orders the reviews shown for each book according to the reviewers most similar to the user. 

## Background and Problem Statement
When reading book reviews before purchasing a book, I often find myself wondering how much weight I should give the reviews. Was the reviewer similar to me and if their experience therefore would be similar to mine? I also never really read past the first 5 reviews, which are usually ranked by reviewer ranking or the number of likes. This gave me the idea to build a book review recommender system where I would first build a book recommender system and use this information to find the optimal order the reviews should appear in for every user and their recommended books, where the reviews are in order of the most similar reviewers to them. 

While this project would just consider books, this could extend to other products where the experience vs the functionality of the product would be more important. Given the overwhelming number of options for each product with the rise of e-commerce, more and more customers are starting to put increasing weight on reviews. Currently Amazon and Goodreads order their reviews by the reviewer ranking or the number of likes/helpful votes a post gets but this is not personalised to users and in the age of recommender systems, feels fitting that it should be the next step. Therefore, this feature could be an interesting addition and actually is a requested feature on GoodReads currently. 

## Data Accquring & Cleaning
Notebooks: `data_cleaning` & `connecting_to_gcloud`

I am using the the [datasets provided by Julian McAuley and his team at USCD](http://jmcauley.ucsd.edu/data/amazon/), in particular the 2014 5-core book review dataset and the book metadata. 

The data was presented in large JSON files that I processed and cleaned chunkwise, given the size of the files and capablities of my laptop, before storing them as CSV files. The cleaning processed involved disregarding any incorrect rows with large HTML code from webscrapping, reengineering a few columns for better data storage and so on, which I have documented in the `data_cleaning` notebook. I had initially planned to upload these large CSV files to a Postgres database on GCloud to query from Jupyter Notebooks. But I ended up condensing the dataset and it made more sense to leave them as CSV files locally. However, I did upload the process of connecting to GCloud in the `connecting_to_gcloud` notebook. 

The condensed dataset had ~270k rows with all users and books that had atleast 50 reviews.  

## EDA
Notebook:`eda_book_review` 

I have gone through all of my findings in EDA of the datasets in the notebook `eda_book_review` notebook, but here are the main takeaways. 

The reviews were very skewed to the positive side with mainly 4 or 5 star ratings. 

Negative reviews tend to get more attention as the avg number of total votes (likes) per rating category is the highest for 1 star and 2 star reveiews. This could push negative reviews unfairly to the top. 


Look over the years, from 2000 to 2014, reviewers seem to have gotten more critical over the years with the avg rating dropped from ~4.25 to ~3.9. Note that this across a 4000-day rolling avg as the rating jumped around a lot and needed a large window to smooth the curve. Interestingly, there is a large drop in 2007/2008, around the time of the financial crash, maybe to signify that the emotional toll from the economic collapse split over into how critical the reviewers were? 


These were the top categories of the books 




- EDA: Some EDA. Shows that negative reviews get more attention and other findings. 

## Picking the best algorithm and Gridsearching (Hypertuning)
Notebooks:
- Recommender system and Surprise CV test: The notebook for first checking all the algorithms and then gridsearching the chosen 4. 
- Cosine similarities function script: A script containing a random generator function that allow to pick a user and randomly one of their top 10 books. Function returns the current order of book, recommended order, comparision of categories. 
- Results and metrics: metrics to measure the impact of the model as well as visualising the result of the re-ordering using the cosine similarities function 


## Recommender systems
Notebooks:
I obviously cannot measure the impact of the model as I would need customers to tell me if this reordering is helpful or not but it was interesting to see if the model had any massive impact. 

- Group the categories a bit better
- Use better meta dataset 
- try to include review text context analysis to improve the recommender system 
- neighbourhood models to use for cosine similarity rather than calculating using the whole user-item matrix 


## Results and Next Steps:

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) 

