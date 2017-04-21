
from Aggregators import Aggregators
from Group import Group
from Config import Config
from collections import defaultdict
import numpy as np
import pandas as ps
import warnings
from sklearn.metrics import mean_squared_error

#overflow warnings should be raised as errors
np.seterr(over='raise')

#global class.
class GroupRec:
    def __init__(self):
        self.cfg = Config(r"./config.conf")
        
        #training and testing matrices, init. with random sizes
        self.ratings = np.ndarray((10,10))
        self.test_ratings = np.ndarray((10,10))
        
        #read data into above matrices
        self.read_data()
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]
        
        #predicted ratings matrix based on factors. 
        self.predictions = np.zeros((self.num_users, self.num_items))
        
        #output after self.sgd_factorize()
        #initialize all unknowns with random values from -1 to 1
        self.user_factors = np.random.uniform(-1, 1, (self.ratings.shape[0], self.cfg.num_factors))
        self.item_factors = np.random.uniform(-1, 1, (self.ratings.shape[1], self.cfg.num_factors))
        
        #either above or initialize factors with normally distributed numbers
#         self.user_factors = np.random.normal(scale=1./self.cfg.num_factors, size = (self.num_users, self.cfg.num_factors))
#         self.item_factors = np.random.normal(scale=1./self.cfg.num_factors, size = (self.num_items, self.cfg.num_factors))
        
        self.user_biases = np.zeros(self.num_users)
        self.item_biases = np.zeros(self.num_items)
        
        #global mean of ratings a.k.a mu
        self.ratings_global_mean = 0
        pass

    #add list of groups to grouprec
    def add_groups(self, groups):
        self.groups = groups
        pass
    
    #read training and testing data into matrices
    def read_data(self):
        column_headers = ['user_id', 'item_id', 'rating', 'timestamp']
        print 'Reading training data from ', self.cfg.training_file, '...'
        training_data = ps.read_csv(self.cfg.training_file, sep = '\t', names = column_headers)
        print 'Reading testing data from ', self.cfg.testing_file, '...'
        testing_data = ps.read_csv(self.cfg.testing_file, sep = '\t', names = column_headers)
        
        num_users = max(training_data.user_id.unique())
        num_items = max(training_data.item_id.unique())
        
        self.ratings = np.zeros((num_users, num_items))
        
        for row in training_data.itertuples(index = False):
            self.ratings[row.user_id - 1, row.item_id - 1] = row.rating 
        
    #split data set file into training and test file by ratio 
    def split_data(self, data_file, training_ratio = 0.7):
        pass
    
    def predict_user_rating(self, user, item):
        prediction = self.ratings_global_mean + self.user_biases[user] + self.item_biases[item]
        prediction += self.user_factors[user, :].dot(self.item_factors[item, :].T)
        return prediction
        
    #matrix factorization code, this should be run before af, bf or wbf
    #outputs from this are used in methods
    def sgd_factorize(self):
        #solve for these for matrix ratings        
        ratings_row, ratings_col = self.ratings.nonzero()
        num_ratings = len(ratings_row)
        learning_rate = self.cfg.learning_rate_mf
        regularization = self.cfg.lambda_mf
        
        self.ratings_global_mean = np.mean(self.ratings[np.where(self.ratings != 0)])
        
        print 'Doing matrix factorization...'
        try:
            for iter in range(self.cfg.max_iterations_mf):
                print 'Iteration: ', iter
                rating_indices = np.arange(num_ratings)
                np.random.shuffle(rating_indices)
                
                for idx in rating_indices:
                    user = ratings_row[idx]
                    item = ratings_col[idx]

                    pred = self.predict_user_rating(user, item)
                    error = self.ratings[user][item] - pred
                    
                    self.user_factors[user] += learning_rate \
                                                * ((error * self.item_factors[item]) - (regularization * self.user_factors[user]))
                    self.item_factors[item] += learning_rate \
                                                * ((error * self.user_factors[user]) - (regularization * self.item_factors[item]))
                    
                    self.user_biases[user] += learning_rate * (error - regularization * self.user_biases[user])
                    self.item_biases[item] += learning_rate * (error - regularization * self.item_biases[item])
            
                self.sgd_mse()
            
        except FloatingPointError:
            print 'Floating point Error: '
            
    def predict_all_ratings(self):
        for user in range(self.num_users):
            for item in range(self.num_items):
                self.predictions[user, item] = self.predict_user_rating(user, item)
        
    
    def sgd_mse(self):
        self.predict_all_ratings()
        predicted_ratings = self.predictions[self.ratings.nonzero()].flatten()
        actual_ratings = self.ratings[self.ratings.nonzero()].flatten()
    
        mse = mean_squared_error(predicted_ratings, actual_ratings)
        print 'mse: ', mse
            
        
    #AF method
    # Check if we want this method to take multiple groups or single group
    # as input
    def af_runner(self, groups = None, aggregator = Aggregators.average):
        #if groups is not passed, use self.groups
        if (groups is None):
            groups = self.groups
        
        #calculate factors
        #aggregate the factors
        pass
    
    def bf_runner(self, groups = None, aggregator = Aggregators.average):
        #aggregate user ratings into virtual group
        #calculate factors of group
        pass
        
    def wbf_runner(self, groups = None, aggregator = Aggregators.average):
        pass
    
    # can have this step being done in af_runner/ bf_runner itself
    def evaluation(self):
        pass

if __name__ == "__main__":
    #Workflow
    gr = GroupRec()
    #can, move this function also to config __init__, will decide later
#     gr.read_data()
    #factorize matrix
    gr.sgd_factorize()
    
    #add groups or generate random groups of given size
    groups = []
    members = [1,2,3,4]
    if (Group.can_group(members)):
        Group(members, gr.cfg)
    
    #OR generate groups programmatically
    groups = Group.generate_groups(10, gr.cfg.small_grp_size)
    gr.add_groups(groups)
    
    #PS: could call them without passing groups as we have already added groups to grouprec object
    gr.af_runner(groups, Aggregators.average)
    gr.bf_runner(groups, Aggregators.median)
    gr.wbf_runner(groups, Aggregators.least_misery)
    
    pass

