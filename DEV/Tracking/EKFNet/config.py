
config = {
    "RMS_measurement"        : True,
    "RMS_post"               : True,
    "Likelihood_measurement" : True , 
    "Likelihood_post"        : True, 
    "interval"               : 1,
    "update_rule"            : "adam",
    
    "batch_size"             : 10, 
    "num_epochs"             : 1000, 
    "print_every"            : 1, 
    "verbose"                : True,
    "lr_decay"               : 0.99, 
    'lr'                     : 1e-1, 
    
    # used for unceratinty aware input normalization
    'distance_norm'          : 60.0,
    'n_point_norm'           : 300
}