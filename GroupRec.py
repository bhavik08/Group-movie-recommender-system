from Aggregators import Aggregators
from Group import Group
from Config import Config
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as ps


# overflow warnings should be raised as errors
np.seterr(over='raise')


class GroupRec:
    def __init__(self):
        self.cfg = Config(r"./config.conf")
        
        # training and testing matrices
        self.ratings = None
        self.test_ratings = None

        self.groups = []
        
        # read data into above matrices
        self.read_data()
        
        self.num_users = self.ratings.shape[0]
        self.num_items = self.ratings.shape[1]
        
        # predicted ratings matrix based on factors.
        self.predictions = np.zeros((self.num_users, self.num_items))
        
        # output after svd factorization
        # initialize all unknowns with random values from -1 to 1
        self.user_factors = np.random.uniform(-1, 1, (self.ratings.shape[0], self.cfg.num_factors))
        self.item_factors = np.random.uniform(-1, 1, (self.ratings.shape[1], self.cfg.num_factors))

        self.user_biases = np.zeros(self.num_users)
        self.item_biases = np.zeros(self.num_items)
        
        # global mean of ratings a.k.a mu
        self.ratings_global_mean = 0

    # read training and testing data into matrices
    def read_data(self):
        column_headers = ['user_id', 'item_id', 'rating', 'timestamp']

        print 'Reading training data from ', self.cfg.training_file, '...'
        training_data = ps.read_csv(self.cfg.training_file, sep='\t', names=column_headers)

        print 'Reading testing data from ', self.cfg.testing_file, '...'
        testing_data = ps.read_csv(self.cfg.testing_file, sep='\t', names=column_headers)

        num_users = max(training_data.user_id.unique())
        num_items = max(training_data.item_id.unique())

        self.ratings = np.zeros((num_users, num_items))
        self.test_ratings = np.zeros((num_users, num_items))

        for row in training_data.itertuples(index=False):
            self.ratings[row.user_id - 1, row.item_id - 1] = row.rating

        for row in testing_data.itertuples(index=False):
            self.test_ratings[row.user_id - 1, row.item_id - 1] = row.rating

    # add list of groups
    def add_groups(self, groups):
        self.groups = groups
    
    # remove groups
    def remove_groups(self, groups):
        self.groups = []
    
    def predict_user_rating(self, user, item):
        prediction = self.ratings_global_mean + self.user_biases[user] + self.item_biases[item]
        prediction += self.user_factors[user, :].dot(self.item_factors[item, :].T)
        return prediction
    
    def predict_group_rating(self, grp, item, method):
        bias_grp = 0
        factors = np.nan
        if method == 'af':
            factors = grp.grp_factors_af; bias_grp = grp.bias_af
        elif method == 'bf':
            factors = grp.grp_factors_bf; bias_grp = grp.bias_bf
        elif method == 'wbf':
            factors = grp.grp_factors_wbf; bias_grp = grp.bias_wbf
        
        return self.ratings_global_mean + bias_grp + self.item_biases[item] \
                                        + np.dot(factors.T, self.item_factors[item])

    def predict_all_ratings(self):
        for user in range(self.num_users):
            for item in range(self.num_items):
                self.predictions[user, item] = self.predict_user_rating(user, item)
        
    # matrix factorization code
    def sgd_factorize(self):
        regularization = self.cfg.lambda_mf
        learning_rate = self.cfg.learning_rate_mf

        self.ratings_global_mean = np.mean(self.ratings[np.where(self.ratings != 0)])

        # solve for these for matrix ratings
        ratings_row, ratings_col = self.ratings.nonzero()
        num_ratings = len(ratings_row)
        
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
                    
                    self.user_factors[user] += learning_rate * ((error * self.item_factors[item])
                                                    - (regularization * self.user_factors[user]))
                    self.item_factors[item] += learning_rate * ((error * self.user_factors[user])
                                                    - (regularization * self.item_factors[item]))
                    
                    self.user_biases[user] += learning_rate * (error - regularization * self.user_biases[user])
                    self.item_biases[item] += learning_rate * (error - regularization * self.item_biases[item])
            
                if self.cfg.is_debug:
                    self.sgd_mse()
            
        except FloatingPointError:
            print 'Floating point Error: '

    def sgd_mse(self):
        self.predict_all_ratings()
        predicted_training_ratings = self.predictions[self.ratings.nonzero()].flatten()
        actual_training_ratings = self.ratings[self.ratings.nonzero()].flatten()
        
        predicted_test_ratings = self.predictions[self.test_ratings.nonzero()].flatten()
        actual_test_ratings = self.test_ratings[self.test_ratings.nonzero()].flatten()
    
        training_mse = mean_squared_error(predicted_training_ratings, actual_training_ratings)
        print 'training mse: ', training_mse

        test_mse = mean_squared_error(predicted_test_ratings, actual_test_ratings)
        print 'test mse: ', test_mse

    # AF Method
    def af_runner(self, grps=None, aggregator=Aggregators.average):
        if grps is None:
            grps = self.groups
        
        # calculate factors
        for grp in grps:
            member_factors = self.user_factors[grp.members, :]
            member_biases = self.user_biases[grp.members]
        
            # aggregate the factors
            if aggregator == Aggregators.average:
                grp.grp_factors_af = aggregator(member_factors)
                grp.bias_af = aggregator(member_biases)
            elif aggregator == Aggregators.weighted_average:
                grp.grp_factors_af = aggregator(member_factors, weights=grp.ratings_per_member)
                grp.bias_af = aggregator(member_biases, weights=grp.ratings_per_member)

            self.make_recommendations_for_af(grp)

    def make_recommendations_for_af(self, grp):
        # predict ratings for all candidate items
        group_candidate_ratings = {}
        for idx, item in enumerate(grp.candidate_items):
            cur_rating = self.predict_group_rating(grp, item, 'af')
            if cur_rating > self.cfg.rating_threshold_af:
                group_candidate_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_af' recommendations
        group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)[:self.cfg.num_recos_af]
        grp.reco_list_af = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])

    def make_recommendations_for_bf(self, grp):
        # predict ratings for all candidate items
        group_candidate_ratings = {}
        for idx, item in enumerate(grp.candidate_items):
            cur_rating = self.predict_group_rating(grp, item, 'bf')
            if cur_rating > self.cfg.rating_threshold_bf:
                group_candidate_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_bf' recommendations
        group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)[:self.cfg.num_recos_bf]
        grp.reco_list_bf = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])

    def make_recommendations_for_wbf(self, grp):
        # predict ratings for all candidate items
        group_candidate_ratings = {}
        for idx, item in enumerate(grp.candidate_items):
            cur_rating = self.predict_group_rating(grp, item, 'wbf')
            if cur_rating > self.cfg.rating_threshold_wbf:
                group_candidate_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_wbf' recommendations
        group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)[:self.cfg.num_recos_wbf]
        grp.reco_list_wbf = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])

    def get_weight_matrix(self, grp, watched_items):
        wt = []
        for item in watched_items:
            rated = np.argwhere(self.ratings[:, item] != 0)  # list of users who have rated this movie
            watched = np.intersect1d(rated, grp)  # list of group members who have watched this movie
            std_dev = np.std(filter(lambda a: a != 0, self.ratings[:, item]))  # std deviation for the rating of the item
            wt += [len(watched) / float(len(grp.members)) * 1 / (1 + std_dev)]  # list containing diagonal elements
        W = np.diag(wt)  # diagonal weight matrix
        return W

    # BF/WBF Method
    def bf_runner(self, grps=None, aggregator=Aggregators.average_bf, is_wbf=False):
        # aggregate user ratings into virtual group
        # calculate factors of group
        lamb = self.cfg.lambda_mf
        for grp in grps:
            all_movies = np.arange(len(self.ratings.T))
            watched_items = sorted(list(set(all_movies) - set(grp.candidate_items)))

            group_rating = self.ratings[grp.members, :]
            agg_rating = aggregator(group_rating)
            s_g = []
            for j in watched_items:
                s_g.append(agg_rating[j] - self.ratings_global_mean - self.item_biases[j])

            # creating matrix A : contains rows of [item_factors of items in watched_list + '1' vector]
            A = np.zeros((0, self.cfg.num_factors))  # 3 is the number of features here = K

            for item in watched_items:
                A = np.vstack([A, self.item_factors[item]])
            v = np.ones((len(watched_items), 1))
            A = np.c_[A, v]

            if is_wbf:
                W = self.get_weight_matrix(grp, watched_items)
                factor_n_bias = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W),A) + lamb * np.identity(self.cfg.num_factors + 1)), np.dot(np.dot(A.T, W), s_g))
                grp.grp_factors_wbf = factor_n_bias[:-1]
                grp.bias_wbf = factor_n_bias[-1]
                self.make_recommendations_for_wbf(grp)
            else:
                factor_n_bias = np.dot(np.linalg.inv(np.dot(A.T, A) + lamb * np.identity(self.cfg.num_factors + 1)), np.dot(A.T, s_g))
                grp.grp_factors_bf = factor_n_bias[:-1]
                grp.bias_bf = factor_n_bias[-1]
                self.make_recommendations_for_bf(grp)

    def evaluation(self):
        # For AF
        af_precision_list = []
        af_recall_list = []
        print "\n#########-------For AF-------#########"
        for grp in self.groups:
            grp.generate_actual_recommendations(self.test_ratings, self.cfg.rating_threshold_af, self.cfg.is_debug)
            (precision, recall, tp, fp) = grp.evaluate_af(self.cfg.is_debug)
            af_precision_list.append(precision)
            af_recall_list.append(recall)
        
        af_mean_precision = np.nanmean(np.array(af_precision_list))
        af_mean_recall = np.nanmean(np.array(af_recall_list))
        print '\nAF method: mean precision: ', af_mean_precision
        print 'AF method: mean recall: ', af_mean_recall

        # For BF
        bf_precision_list = []
        bf_recall_list = []
        print "\n#########-------For BF-------#########"
        for grp in self.groups:
            grp.generate_actual_recommendations(self.test_ratings, self.cfg.rating_threshold_bf, self.cfg.is_debug)
            (precision, recall, tp, fp) = grp.evaluate_bf(self.cfg.is_debug)
            bf_precision_list.append(precision)
            bf_recall_list.append(recall)

        bf_mean_precision = np.nanmean(np.array(bf_precision_list))
        bf_mean_recall = np.nanmean(np.array(bf_recall_list))
        print '\nBF method: mean precision: ', bf_mean_precision
        print 'BF method: mean recall: ', bf_mean_recall

        # For WBF
        wbf_precision_list = []
        wbf_recall_list = []
        print "\n#########-------For WBF-------#########"
        for grp in self.groups:
            grp.generate_actual_recommendations(self.test_ratings, self.cfg.rating_threshold_wbf, self.cfg.is_debug)
            (precision, recall, tp, fp) = grp.evaluate_wbf(self.cfg.is_debug)
            wbf_precision_list.append(precision)
            wbf_recall_list.append(recall)

        wbf_mean_precision = np.nanmean(np.array(wbf_precision_list))
        wbf_mean_recall = np.nanmean(np.array(wbf_recall_list))
        print '\nWBF method: mean precision: ', wbf_mean_precision
        print 'WBF method: mean recall: ', wbf_mean_recall

    def run_all_methods(self, grps):
        if grps is None:
            grps = self.groups

        self.af_runner(grps, Aggregators.weighted_average)
        self.bf_runner(grps, Aggregators.average_bf)
        self.bf_runner(grps, Aggregators.average_bf, is_wbf=True)  # For WBF

        # evaluation
        self.evaluation()
    

if __name__ == "__main__":
    gr = GroupRec()

    # factorize matrix
    gr.sgd_factorize()
    
    # add groups or generate random groups of given size
    groups = []

    # members = [475, 549, 775]
    # candidate_items = Group.find_candidate_items(gr.ratings, members)
    # if len(candidate_items) != 0:
    #   groups = [Group(gr.cfg, members, candidate_items, gr.ratings)]

    # disjoint means none of the groups shares any common members
    small_groups = Group.generate_groups(gr.ratings, gr.test_ratings, gr.num_users, gr.cfg.no_of_small_grps, gr.cfg.small_grp_size, disjoint=False)
    medium_groups = Group.generate_groups(gr.ratings, gr.test_ratings, gr.num_users, gr.cfg.no_of_medium_grps, gr.cfg.medium_grp_size, disjoint=False)
    large_groups = Group.generate_groups(gr.ratings, gr.test_ratings, gr.num_users, gr.cfg.no_of_large_grps, gr.cfg.large_grp_size, disjoint=False)
    
    group_set = [small_groups, medium_groups, large_groups]
    group_type = ['small', 'medium', 'large']
    
    for idx, groups in enumerate(group_set):
        if groups is []:
            continue
        
        # generated groups
        n = len(groups) if gr.cfg.is_debug else 5
        print '\n******* Running for ', group_type[idx], ' groups *************'
        print 'generated groups (only %d are getting printed here): ' % n
        for group in groups[:n]:
            print(group.members)
        
        gr.add_groups(groups)
        gr.run_all_methods(groups)
        gr.remove_groups(groups)
