
config = {
    "RMS_measurement"        : False,
    "RMS_post"               : True,
    "Likelihood_measurement" : False , 
    "Likelihood_post"        : False, 
    "interval"               : 50,
    "update_rule"            : "adam",
    
    "batch_size"             : 10, 
    "num_epochs"             : 1000, 
    "print_every"            : 1, 
    "verbose"                : True,
    "lr_decay"               : 0.99, 
    'lr'                     : 1e-1
}