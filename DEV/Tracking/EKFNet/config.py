
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
    'lr'                     : 1e-1
}