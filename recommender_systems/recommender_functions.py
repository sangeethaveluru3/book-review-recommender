import random
import pandas as pd
import numpy as np
import surprise as sur
import scipy.spatial.distance as sc
from collections import defaultdict
from scipy.stats.stats import pearsonr





def get_top_10(pred_matrix_, df_):
    """ Returns the top 10 recommendations for each user
    
    Args: 
    - Prediction matrix in a dataframe: the rows should be users & columns should be books
    - The original dataframe with known ratings
    
    Returns:
    - A dataframe of n recommendations for every user: cols = user, book, estimated rating
    """
    
    top_10_list=[]
    
    copy_matrix = pred_matrix_.copy()
    
    #Set all of the known ratings to nan as these should not be recommended again
    for val in df_.values:
        copy_matrix.loc[val[0], val[1]] = np.nan
    
    for user in copy_matrix.index:
        
        #get the top 15 items
        top_15 = copy_matrix.loc[user].sort_values(ascending = False)[:15]

        #Choose 15, pick the first 5 and sample the next 10 for 5 samples
        #This improves the coverage ratio and makes sure a variety of books are being recommended
        top_15_list= list(zip(top_15.index, top_15.values))
        recos = top_15_list[:5]
        recos += random.sample(top_15_list[5:], 5)
        
        #add to list to get it in the desired format
        for val in recos:
            top_10_list.append([user, val[0], val[1]])
    
    return pd.DataFrame(top_10_list, columns = ['user', 'book', 'est_rating'])





def calculate_coverage(top_10_recos_, df_):
    """Returns coverage ratio of the algorithm, essentially the number of unique books in the top 10
    recommendations as a fraction of the whole book base
    
    Args:
    - A dataframe of n recommendations for every user: cols = user, book, estimated rating
    - The original dataframe with known ratings: cols = user, book, estimated rating
    
    Returns:
    - Coverage ratio (float)
    """
    return len(top_10_recos_.book.unique())/len(df_.asin.unique())





def precision_recall_at_k(predictions, k=10, threshold=4.1):
    """Returns precision and recall at k metrics for each user.
    
    Args:
    - predictions train (as outputted by the test method in Surprise)
    - k (int)
    - threshold value (int/float)
    
    Return:
    - precision@k
    - recall@k 
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls





def calculate_impact(top_10_recos_, df_, pred_matrix_):
    """ Calculates the 3 different ways of acessing the impact of reordering 
    the reviews according to user similarities for each user-book pair:
    
        1) rating difference: difference sum of the first 10 reviews before and after reordering 
        
        2) rank difference: difference in the sum of the product of rating and rank the review is shows 
                            before and after reordering
        
        3) Pearson correlation of the ranks before and after (essentially Spearman correlation of the ratings 
        before and after)
        
    Args:
    - Top 10 recommendations dataframe: cols = user, book, estimated rating
    - Original book review dataframe: cols = user, book, known rating 
    - Prediction matrix: cols = rows are users and columns are books
    
    Returns:
    - A dataframe containing all three calculations: cols = user, book, estimated rating, rating difference, 
                                                            ranking difference and correlation
    """
    
    impact = []
    
    #the counter is just for debugging purposes, remove if not necessary
    counter = 0

    for row in top_10_recos_.values:
        
        #for user and book pair, first obtain the existing reviews for the book
        relevant_reviews = df_[df_.asin == row[1]].copy()

        #for each reviewer in the reviewers who have rated the book calculate the cosine similarity 
        #between that reviewer and the user in question
        distances_ = []

        for reviewer_ in relevant_reviews.reviewerId:
            distances_.append(
                    sc.cosine(pred_matrix_.loc[row[0], :].values, pred_matrix_.loc[reviewer_, :].values))

        relevant_reviews['cos_dist'] = distances_

        #get the ranking by total_votes: i.e. the order in which the reviews would currently be displayed
        relevant_reviews['vote_rank'] = relevant_reviews['total_votes'].rank(
                method='average', ascending=False)

        #get the ranking by cosine distances : i.e. the new order of the reviews according to similarities
        relevant_reviews['cos_rank']= relevant_reviews['cos_dist'].rank(
                method='dense', ascending=False)
        
        #calculate the the correlation between the old and new order of the reviews 
        #Note: if there is no change in the order, this would throw an error, hence the try/except statements
        try:
            spearman = pearsonr(relevant_reviews['vote_rank'], relevant_reviews['cos_rank'])[1]

        except:
            spearman = np.nan

        #calculate difference in the first 10 rating before and after the reordering
        rating_diff = relevant_reviews.sort_values(
            by='total_votes', ascending=False)['rating'][:10].sum() - relevant_reviews.sort_values(
                by='cos_dist', ascending=True)['rating'][:10].sum()
        
        # Calculate the difference in rank*rating before and after the reordering
        relevant_reviews['vote_rank'] = relevant_reviews['vote_rank']*relevant_reviews['rating']
        relevant_reviews['cos_rank'] = relevant_reviews['cos_rank']*relevant_reviews['rating']

        ranking_diff = relevant_reviews['vote_rank'].sum() - relevant_reviews['cos_rank'].sum()
        
        #Add the values to the impact list
        impact.append((row[0], row[1], row[2], rating_diff, ranking_diff, spearman))

        counter += 1
        print(counter)
    
    return pd.DataFrame(impact, columns = ['reviewerId', 'asin', 'est_rating', 
                                              'rating_diff', 'weighted_ranking_diff', 'spearman_corr'])


def get_categories(user_, top_10_recos_, merged_):
    '''Returns top 10 categories previously read by the user and top 10 categories of the recommended books
    
    Args:
    - User
    - Top 10 recommendations dataframe: cols = user, book, estimated rating
    - The merged (and exploded - in terms of categories) dataframe of books and metadata
    
    Returns: 
    - The top 10 categories previously read by the user 
    - The top 10 categories of the recommended books
    '''
    
    #Top 10 categories previously read
    read_cat_ = merged_[merged_.reviewerId == user_].categories.value_counts(normalize=True)[:10]
    
    #Top 10 categories recommended
    reco_cat_ = merged[merged.asin.isin(top_10[top_10.user == 'A2RQ0AT4XZUIXL'].book)].categories.value_counts(normalize=True)[:10]
    
    return read_cat_, reco_cat_
    



def review_reorder_example(user_, book_, df_, metadata_, merged_, impact_df_, pred_matrix_):
    """ Returns a visual example of how recordering the reviews would look like i.e. returns the order of reviews
    before and after calculating the user similarities, along with the 3 impact measurement metrics for a chosen
    user-item pair.

    Args:
    - User (reviewerId)
    - Book (asin)
    - Original book review dataframe
    - Book metadata 
    - The merged (and exploded - in terms of categories) dataframe of books and metadata
    - Impact df: the dataframe containing all three calculations of impact
    - Prediction matrix in a dataframe: the rows should be users & columns should be books 

    Returns:
    - Review dataframe (before ordering according to user similarities)
    - The top 10 categories read by the first 5 reviewers before reordering
    - Review dataframe (after ordering according to user similarities)
    - The top 10 categories read by the first 5 reviewers after reordering

    """

    # first obtain the existing reviews for the book
    relevant_reviews = df_[df_.asin == book_].copy()

    relevant_reviews = df_[df_.asin == book].copy()
    relevant_reviews = relevant_reviews.merge(metadata_, on='asin')[
        ['reviewerId', 'title', 'rating', 'summary', 'total_votes', 'categories']]

    distances_ = []

    # Calculate the cosine similarities for this user and all reviewers who have reviewed the book
    for reviewer_ in relevant_reviews.reviewerId:
        distances_.append(
            sc.cosine(pred_matrix_.loc[user_, :].values, pred_matrix_.loc[reviewer_, :].values))

    relevant_reviews['cos_dist'] = distances_

    # Get the ids of the top 5 reviewers before and after reviewing
    top_5_reviewers_before = relevant_reviews.sort_values(
        by='total_votes', ascending=False).reviewerId[:5]
    top_5_reviewers_after = relevant_reviews.sort_values(
        by='cos_dist').reviewerId[:5]

    # Obtain the top 10 categories of books read for each group
    top_cat_before = merged_[merged_.reviewerId.isin(
        top_5_reviewers_before)].categories.value_counts(normalize=True)[:10]

    top_cat_after = merged_[merged_.reviewerId.isin(
        top_5_reviewers_after)].categories.value_counts(normalize=True)[:10]

    return relevant_reviews.sort_values(by='total_votes', ascending=False), top_cat_before, relevant_reviews.sort_values(by='cos_dist'), top_cat_after