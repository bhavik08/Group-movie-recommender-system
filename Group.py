import numpy as np

class Group():
    def __init__(self, config, members, candidate_items):
        #member ids
        self.members = members
        
        #list of items that can be recommended. These should not have been
        #watched by any member of group
        self.candidate_items = candidate_items
        self.actual_recos = []
        
        #AF
        self.grp_factors_af = []
        self.bias_af = 0
        #eval. metrics for AF method for this group
        self.precision_af = 0
        self.recall_af = 0
        #recommended items acc. to AF method, 
        #after calculatng ratings of candidate items and filtering them
        self.reco_list_af = [] 
        
        #BF
        self.grp_factors_bf = []
        self.bias_bf = 0
        self.precision_bf = 0
        self.recall_bf = 0
        self.reco_list_bf = []
        
        #WBF
        self.grp_factors_wbf = []
        self.bias_wbf = 0
        self.precision_wbf = 0
        self.recall_wbf = 0
        #W matrix from the paper
        self.weight_matrix_wbf = []
        self.reco_list_wbf = []
    
    #verifies that given list of members can form a group w.r.t given data
    #ensure that there is atleast 1 movie in training data that hasn't been 
    #watched by any member of the group. Only these can be recommended.
    #call this method before calling group constructor.
    # For eg.
    @staticmethod
    def find_candidate_items(ratings, members):
        if len(members) == 0: return []
        
        unwatched_items = np.argwhere(ratings[members[0]] == 0)
        for member in members:
            cur_unwatched = np.argwhere(ratings[member] == 0)
            unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)
        
        return unwatched_items
    
    #programmatically generate groups of given size
    @staticmethod
    def generate_groups(cfg, ratings, num_users, count, size, disjoint = True):
        avbl_users = [i for i in range(num_users)]
        groups = []
        
        for iter in range(count):
            group_members = np.random.choice(avbl_users, size = size, replace = False)
            candidate_items = Group.find_candidate_items(ratings, group_members)
            
            if len(candidate_items) != 0:
                groups += [Group(cfg, group_members, candidate_items)]
                avbl_users = np.setdiff1d(avbl_users, group_members)
                
        return groups
    
    def generate_actual_recommendations(self, ratings, threshold):
        items = np.argwhere(ratings[self.members[0]] >= threshold)
        for member in self.members:
            cur_items = np.argwhere(ratings[member] >= threshold)
            items = np.intersect1d(items, cur_items)

        self.actual_recos = items

    def evaluate_af(self):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_af).size)
        self.precision_af = tp / len(self.reco_list_af)
        self.recall_af = tp / self.actual_recos.size

    def evaluate_bf(self):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_bf).size)
        self.precision_bf = tp / len(self.reco_list_bf)
        self.recall_bf = tp / self.actual_recos.size

    def evaluate_wbf(self):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_wbf).size)
        self.precision_wbf = tp / len(self.reco_list_wbf)
        self.recall_wbf = tp / self.actual_recos.size
