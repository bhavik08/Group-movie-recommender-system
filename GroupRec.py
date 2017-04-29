
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
        self.read_data(self.cfg.training_file)
        
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
    def read_data(self, file, is_training = True):
        column_headers = ['user_id', 'item_id', 'rating', 'timestamp']
        print 'Reading data from ', file, '...'
        data = ps.read_csv(file, sep = '\t', names = column_headers)
        #print 'Reading testing data from ', self.cfg.testing_file, '...'
        #testing_data = ps.read_csv(self.cfg.testing_file, sep = '\t', names = column_headers)
        
        if is_training:
            num_users = max(data.user_id.unique())
            num_items = max(data.item_id.unique())
        else:
            num_users = self.num_users
            num_items = self.num_items
        
        self.ratings = np.zeros((num_users, num_items))
        
        for row in data.itertuples(index = False):
            self.ratings[row.user_id - 1, row.item_id - 1] = row.rating 
        
    #split data set file into training and test file by ratio 
    def split_data(self, data_file, training_ratio = 0.7):
        pass
    
    def predict_user_rating(self, user, item):
        prediction = self.ratings_global_mean + self.user_biases[user] + self.item_biases[item]
        prediction += self.user_factors[user, :].dot(self.item_factors[item, :].T)
        return prediction
    
    def predict_group_rating(self, group, item, method):
        #bias_grp and
        if (method == 'af'):
            factors = group.grp_factors_af; bias_group = group.bias_af
        elif (method == 'bf'):
            factors = group.grp_factors_bf; bias_group = group.bias_bf
        elif (method == 'wbf'):
            factors = group.grp_factors_wbf; bias_group = group.bias_wbf
        
        return self.ratings_global_mean + bias_group + self.item_biases[item] \
                                        + np.dot(factors.T, self.item_factors[item])
        
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
        for group in groups:
            member_factors = self.user_factors[group.members, :]
            member_biases = self.user_biases[group.members]
        
            #aggregate the factors
            group.grp_factors_af = aggregator(member_factors)
            group.bias_af = aggregator(member_biases)
            
            #predict ratings for all candidate items
            group_candidate_ratings = np.zeros(group.candidate_items.size)
            for idx, item in enumerate(group.candidate_items):
                group_candidate_ratings[idx] += self.predict_group_rating(group, item, 'af')
            
            #filter to keep top 'num_recos_af' recommendations
            sorted_indices = group_candidate_ratings.argsort()
            group.reco_list_af = group.candidate_items[sorted_indices]
            
            #sorted ratings of recommended items
            group_candidate_ratings = sorted(group_candidate_ratings)
            
            print 'members: ', group.members
            print 'recommended items: ', group.reco_list_af
            print 'recommended item ratings: ', group_candidate_ratings

    def bf_runner(self, groups=None, aggregator=Aggregators.average):
        # aggregate user ratings into virtual group
        # calculate factors of group
        print groups
        lamb = self.cfg.lambda_mf
        for group in groups:
            watched_items = np.argwhere(self.ratings[group.members[0]] != 0).flatten()  # movies rated by first member
            for member in group.members:
                cur_watched = np.argwhere(self.ratings[member] != 0)
                watched_items = np.union1d(watched_items, cur_watched.flatten())

            s_g = []
            for j in range(len(watched_items)):
                s_g.append(watched_items[j] - self.ratings_global_mean - self.item_biases[item])
            print s_g

            # creating matrix A : contains rows of [item_factors of items in watched_list + '1' vector]
            A = np.zeros((0, 3))  # 3 is the number of features here = K

            for item in watched_items:
                A = np.vstack([A, self.item_factors[item]])
            v = np.ones((len(watched_items), 1))
            A = np.c_[A, v]
            print A
            factor_n_bias = (np.linalg.inv((A.T) * A + lamb * np.identity(3 + 1))) * A.T * s_g
            group_factor = factor_n_bias[:-1]
            group_bias = factor_n_bias[-1]

            # Making recommendations on candidate list :
            predicted_items = []
            for item in self.find_candidate_items:
                # calculating predicted score :
                pred = self.ratings_global_mean + self.item_biases[item] + group_bias + (group_factor).T * \
                                                                                        self.item_factors[item]
                predicted_items.append(pred)
            # returning top 50 items from predicted list
            print 'We believe', group, 'will enjoy these movies!', sorted(predicted_items)[:50]
        
    def wbf_runner(self, groups = None, aggregator = Aggregators.average):
        pass

    def evaluation(self):
        self.read_data(self.cfg.testing_file, False)

        # For AF
        for grp in self.groups:
            grp.generate_actual_recommendations(self.ratings, self.cfg.rating_threshold_af)
            grp.evaluate_af()

        # For BF
        for grp in self.groups:
            grp.generate_actual_recommendations(self.ratings, self.cfg.rating_threshold_bf)
            grp.evaluate_bf()

        # For WBF
        for grp in self.groups:
            grp.generate_actual_recommendations(self.ratings, self.cfg.rating_threshold_wbf)
            grp.evaluate_wbf()


if __name__ == "__main__":
    #Workflow
    gr = GroupRec()
    #can, move this function also to config __init__, will decide later
#     gr.read_data()
    #factorize matrix
    gr.sgd_factorize()
    
    #add groups or generate random groups of given size
    groups = []
    members = [475, 549, 775]
    candidate_items = Group.find_candidate_items(gr.ratings, members)
    if len(candidate_items) != 0:
        groups = [Group(gr.cfg, members, candidate_items)]
    
    #OR generate groups programmatically
    #disjoint means none of the groups shares any common members     
    #groups = Group.generate_groups(gr.cfg, gr.ratings, gr.num_users, 10, gr.cfg.small_grp_size, disjoint=True)
    gr.add_groups(groups)
    
    #generated groups
    print 'generated groups: '
    for group in groups:
        print(group.members)
    
    #PS: could call them without passing groups as we have already added groups to grouprec object
    gr.af_runner(groups, Aggregators.average)
    gr.bf_runner(groups, Aggregators.median)
    gr.wbf_runner(groups, Aggregators.least_misery)

    #evaluation
    gr.evaluation()
    pass

