
class Group():
    def __init__(self, config, members):
        #member ids
        self.members = []
        
        #list of items that can be recommended. These should not have been
        #watched by any member of group
        self.candidate_items = []
        
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
    # if (can_group(members): Group(members)
    @staticmethod
    def can_group(self, members):
        return True
    
    #programmatically generate groups of given size
    def generate_groups(self, count, size):
        pass
    
    #iterate over the rating matrix and find candidate items for this group
    def find_candidate_items(self):
        pass
    
    #given bias
    def predict_rating(self, mean, bias_item, method, item_factors):
        #bias_grp and
        if (method == 'af'):
            factors = self.grp_factors_af; bias_group = self.bias_af
        elif (method == 'bf'):
            factors = self.grp_factors_bf; bias_group = self.bias_bf
        elif (method == 'wbf'):
            factors = self.grp_factors_wbf; bias_group = self.bias_wbf
        
        #unclear
        return mean + bias_item + bias_group + sum([factors[i]*item_factors[i] for i in range(len(factors))])
        pass
    
    
    