import ConfigParser


class Config:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

        configParser = ConfigParser.RawConfigParser()
        configParser.read(config_file_path)
        
        # movie lens 100k dataset, 80 - 20 train/test ratio, present in data directory
        self.training_file = configParser.get('Config', 'training_file')
        self.testing_file = configParser.get('Config', 'testing_file')
        
        self.small_grp_size = int(configParser.get('Config', 'small_grp_size'))
        self.medium_grp_size = int(configParser.get('Config', 'medium_grp_size'))
        self.large_grp_size = int(configParser.get('Config', 'large_grp_size'))

        self.no_of_small_grps = int(configParser.get('Config', 'no_of_small_grps'))
        self.no_of_medium_grps = int(configParser.get('Config', 'no_of_medium_grps'))
        self.no_of_large_grps = int(configParser.get('Config', 'no_of_large_grps'))
        
        self.max_iterations_mf = int(configParser.get('Config', 'max_iterations_mf'))
        self.lambda_mf = float(configParser.get('Config', 'lambda_mf'))
        self.learning_rate_mf = float(configParser.get('Config', 'learning_rate_mf'))
        
        self.num_factors = int(configParser.get('Config', 'num_factors'))
        
        # AF (after factorization)
        self.rating_threshold_af = float(configParser.get('Config', 'rating_threshold_af'))
        self.num_recos_af = int(configParser.get('Config', 'num_recos_af'))
        
        # BF (before factorization)
        self.rating_threshold_bf = float(configParser.get('Config', 'rating_threshold_bf'))
        self.num_recos_bf = int(configParser.get('Config', 'num_recos_bf'))
        
        # WBF (weighted before factorization)
        self.rating_threshold_wbf = float(configParser.get('Config', 'rating_threshold_wbf'))
        self.num_recos_wbf = int(configParser.get('Config', 'num_recos_wbf'))

        self.is_debug = configParser.getboolean('Config', 'is_debug')
