import numpy as np


class Group:
    def __init__(self, members, candidate_items, ratings):
        # member ids
        self.members = sorted(members)
        
        # List of items that can be recommended.
        # These should not have been watched by any member of group.
        self.candidate_items = candidate_items

        self.actual_recos = []
        self.false_positive = []
        
        self.ratings_per_member = [np.size(ratings[member].nonzero()) for member in self.members]
        
        # AF
        self.grp_factors_af = []
        self.bias_af = 0
        self.precision_af = 0
        self.recall_af = 0
        self.reco_list_af = [] 
        
        # BF
        self.grp_factors_bf = []
        self.bias_bf = 0
        self.precision_bf = 0
        self.recall_bf = 0
        self.reco_list_bf = []
        
        # WBF
        self.grp_factors_wbf = []
        self.bias_wbf = 0
        self.precision_wbf = 0
        self.recall_wbf = 0
        self.weight_matrix_wbf = []
        self.reco_list_wbf = []
    
    # Verifies that given list of members can form a group w.r.t given data
    # ensure that there is at least 1 movie in training data that hasn't been
    # watched by any member of the group. Only these can be recommended.
    # call this method before calling group constructor.
    @staticmethod
    def find_candidate_items(ratings, members):
        if len(members) == 0:
            return []
        
        unwatched_items = np.argwhere(ratings[members[0]] == 0)
        for member in members:
            cur_unwatched = np.argwhere(ratings[member] == 0)
            unwatched_items = np.intersect1d(unwatched_items, cur_unwatched)
        
        return unwatched_items
    
    # generate groups of given size
    @staticmethod
    def generate_groups(ratings, test_ratings, num_users, count, size, disjoint=True):
        avbl_users = [i for i in range(num_users)]
        groups = []
        testable_threshold = 50
        
        iter_idx = 0
        while iter_idx in range(count):
            group_members = np.random.choice(avbl_users, size=size, replace=False)
            candidate_items = Group.find_candidate_items(ratings, group_members)
            non_eval_items = Group.non_testable_items(group_members, test_ratings)
            testable_items = np.setdiff1d(candidate_items, non_eval_items)
            
            if len(candidate_items) != 0 and len(testable_items) >= testable_threshold:
                groups += [Group(group_members, candidate_items, ratings)]

                if disjoint:
                    avbl_users = np.setdiff1d(avbl_users, group_members)
                iter_idx += 1
                
        return groups
    
    @staticmethod
    def non_testable_items(members, ratings): 
        non_eval_items = np.argwhere(ratings[members[0]] == 0)
        for member in members:
            cur_non_eval_items = np.argwhere(ratings[member] == 0)
            non_eval_items = np.intersect1d(non_eval_items, cur_non_eval_items)
        return non_eval_items
    
    def generate_actual_recommendations(self, ratings, threshold, is_debug=False):
        non_eval_items = Group.non_testable_items(self.members, ratings)
            
        items = np.argwhere(np.logical_or(ratings[self.members[0]] >= threshold, ratings[self.members[0]] == 0)).flatten()
        fp = np.argwhere(np.logical_and(ratings[self.members[0]] > 0, ratings[self.members[0]] < threshold)).flatten()
        for member in self.members:
            cur_items = np.argwhere(np.logical_or(ratings[member] >= threshold, ratings[member] == 0)).flatten()
            fp = np.union1d(fp, np.argwhere(np.logical_and(ratings[member] > 0, ratings[member] < threshold)).flatten())
            items = np.intersect1d(items, cur_items)
        
        items = np.setdiff1d(items, non_eval_items)

        self.actual_recos = items
        self.false_positive = fp
        if is_debug:
            print '\nT: ', self.actual_recos.size, ' F: ', self.false_positive.size

    def evaluate_af(self, is_debug=False):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_af).size)
        fp = float(np.intersect1d(self.false_positive, self.reco_list_af).size)
        
        try:
            self.precision_af = tp / (tp + fp)
        except ZeroDivisionError:
            self.precision_af = np.NaN

        try:
            self.recall_af = tp / self.actual_recos.size
        except ZeroDivisionError:
            self.recall_af = np.NaN

        if is_debug:
            print 'tp: ', tp
            print 'fp: ', fp
            print 'precision_af: ', self.precision_af
            print 'recall_af: ', self.recall_af

        return self.precision_af, self.recall_af, tp, fp

    def evaluate_bf(self, is_debug=False):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_bf).size)
        fp = float(np.intersect1d(self.false_positive, self.reco_list_bf).size)

        try:
            self.precision_bf = tp / (tp + fp)
        except ZeroDivisionError:
            self.precision_bf = np.NaN

        try:
            self.recall_bf = tp / self.actual_recos.size
        except ZeroDivisionError:
            self.recall_bf = np.NaN

        if is_debug:
            print 'tp: ', tp
            print 'fp: ', fp
            print 'precision_bf: ', self.precision_bf
            print 'recall_bf: ', self.recall_bf

        return self.precision_bf, self.recall_bf, tp, fp

    def evaluate_wbf(self, is_debug=False):
        tp = float(np.intersect1d(self.actual_recos, self.reco_list_wbf).size)
        fp = float(np.intersect1d(self.false_positive, self.reco_list_wbf).size)

        try:
            self.precision_wbf = tp / (tp + fp)
        except ZeroDivisionError:
            self.precision_wbf = np.NaN

        try:
            self.recall_wbf = tp / self.actual_recos.size
        except ZeroDivisionError:
            self.recall_wbf = np.NaN

        if is_debug:
            print 'tp: ', tp
            print 'fp: ', fp
            print 'precision_bf: ', self.precision_wbf
            print 'recall_bf: ', self.recall_wbf

        return self.precision_wbf, self.recall_wbf, tp, fp
