
from Aggregators import Aggregators
from Group import Group
from Config import Config
from collections import defaultdict
import numpy as np
import pandas as ps

#global class.
class GroupRec:
    def __init__(self):
        self.cfg = Config(r"./config.conf")
        #training and testing rating matrices, rows: users, cols: items
        #shall we use dict?
        #maybe Numpy arrays.? think about it. code so that it can be changed midway
        #without affecting much.
        #many row and column operations have simple APIs in Numpy
        self.M_train = np.array([10, 10])
        self.M_test = np.array([10, 10])
        
        #output after self.sgd_factorize()
        self.item_factors = []
        self.user_factors = []
        self.item_biases = []
        self.user_biases = []
        
        pass

    #add list of groups to grouprec
    def add_groups(self, groups):
        self.groups = groups
        pass
    
    #read training and testing data into matrices
    def read_data(self):
        column_headers = ['user_id', 'item_id', 'rating', 'timestamp']
        training_data = ps.read_csv(self.cfg.training_file, sep = '\t', names = column_headers)
        testing_data = ps.read_csv(self.cfg.testing_file, sep = '\t', names = column_headers)
        
        num_users = max(training_data.user_id.unique())
        num_items = max(training_data.item_id.unique())
        
        self.M_train = np.zeros((num_users, num_items))
        
        for row in training_data.itertuples(index = False):
            self.M_train[row.user_id - 1, row.item_id - 1] = row.rating 
        print self.M_train
        
    #split data set file into training and test file by ratio 
    def split_data(self, data_file, training_ratio = 0.7):
        pass
    
    #matrix factorization code, this should be run before af, bf or wbf
    #outputs from this are used in methods
    def sgd_factorize(self):
        #solve for these fors matrix M_train
        self.item_factors = []
        self.user_factors = []
        self.item_biases = []
        self.user_biases = []
        pass
    
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
    gr.read_data()
    #factorize matrix
    gr.sgd_factorize()
    
    #add groups or generate random groups of given size
    groups = []
    members = [1,2,3,4]
    if (Group.can_group(members)):
        Group(members, gr.cfg)
    
    #OR
    groups = Group.generate_groups(10, gr.cfg.small_grp_size)
    gr.add_groups(groups)
    
    #PS: could call them without passing groups as we have already added groups to grouprec object
    gr.af_runner(groups, Aggregators.average)
    gr.bf_runner(groups, Aggregators.median)
    gr.wbf_runner(groups, Aggregators.least_misery)
    
    pass

