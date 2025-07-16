import yaml

# Function to define EQ filters (example)
def define_eq_filters(
    l1_low_shelf_gain_db=0.0, l1_low_shelf_cutoff_freq=100.0, l1_low_shelf_q_factor=0.7,
    l1_band0_gain_db=0.0, l1_band0_cutoff_freq=500.0, l1_band0_q_factor=1.0,
    l1_band1_gain_db=0.0, l1_band1_cutoff_freq=1000.0, l1_band1_q_factor=1.0,
    l1_band2_gain_db=0.0, l1_band2_cutoff_freq=2000.0, l1_band2_q_factor=1.0,
    l1_band3_gain_db=0.0, l1_band3_cutoff_freq=4000.0, l1_band3_q_factor=1.0,
    l1_high_shelf_gain_db=0.0, l1_high_shelf_cutoff_freq=8000.0, l1_high_shelf_q_factor=0.7,
    l2_low_shelf_gain_db=0.0, l2_low_shelf_cutoff_freq=100.0, l2_low_shelf_q_factor=0.7,
    l2_band0_gain_db=0.0, l2_band0_cutoff_freq=500.0, l2_band0_q_factor=1.0,
    l2_band1_gain_db=0.0, l2_band1_cutoff_freq=1000.0, l2_band1_q_factor=1.0,
    l2_band2_gain_db=0.0, l2_band2_cutoff_freq=2000.0, l2_band2_q_factor=1.0,
    l2_band3_gain_db=0.0, l2_band3_cutoff_freq=4000.0, l2_band3_q_factor=1.0,
    l2_high_shelf_gain_db=0.0, l2_high_shelf_cutoff_freq=8000.0, l2_high_shelf_q_factor=0.7
):
    """
    Example function to process EQ parameters for L1 and L2 in a Wiener-Hammerstein model.
    Prints parameters for demonstration; in practice, compute biquad coefficients or apply filters.
    """
    print("L1 Parameters:")
    print(f"  Low Shelf: gain={l1_low_shelf_gain_db} dB, freq={l1_low_shelf_cutoff_freq} Hz, Q={l1_low_shelf_q_factor}")
    for i in range(4):
        print(f"  Band {i}: gain={locals()[f'l1_band{i}_gain_db']} dB, "
              f"freq={locals()[f'l1_band{i}_cutoff_freq']} Hz, Q={locals()[f'l1_band{i}_q_factor']}")
    print(f"  High Shelf: gain={l1_high_shelf_gain_db} dB, freq={l1_high_shelf_cutoff_freq} Hz, Q={l1_high_shelf_q_factor}")
    
    print("\nL2 Parameters:")
    print(f"  Low Shelf: gain={l2_low_shelf_gain_db} dB, freq={l2_low_shelf_cutoff_freq} Hz, Q={l2_low_shelf_q_factor}")
    for i in range(4):
        print(f"  Band {i}: gain={locals()[f'l2_band{i}_gain_db']} dB, "
              f"freq={locals()[f'l2_band{i}_cutoff_freq']} Hz, Q={locals()[f'l2_band{i}_q_factor']}")
    print(f"  High Shelf: gain={l2_high_shelf_gain_db} dB, freq={l2_high_shelf_cutoff_freq} Hz, Q={l2_high_shelf_q_factor}")
    
    # Example: Return a dictionary or process further (e.g., compute biquad coefficients)
    return {
        'l1': {'low_shelf': [l1_low_shelf_gain_db, l1_low_shelf_cutoff_freq, l1_low_shelf_q_factor],
               'bands': [[locals()[f'l1_band{i}_gain_db'], locals()[f'l1_band{i}_cutoff_freq'], locals()[f'l1_band{i}_q_factor']] for i in range(4)],
               'high_shelf': [l1_high_shelf_gain_db, l1_high_shelf_cutoff_freq, l1_high_shelf_q_factor]},
        'l2': {'low_shelf': [l2_low_shelf_gain_db, l2_low_shelf_cutoff_freq, l2_low_shelf_q_factor],
               'bands': [[locals()[f'l2_band{i}_gain_db'], locals()[f'l2_band{i}_cutoff_freq'], locals()[f'l2_band{i}_q_factor']] for i in range(4)],
               'high_shelf': [l2_high_shelf_gain_db, l2_high_shelf_cutoff_freq, l2_high_shelf_q_factor]}
    }

# Read YAML file and convert to dictionary
def load_eq_parameters(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Flatten the nested dictionary for keyword arguments
        flat_params = {}
        
        # Process L1 parameters
        flat_params['l1_low_shelf_gain_db'] = config['L1']['low_shelf']['gain_db']
        flat_params['l1_low_shelf_cutoff_freq'] = config['L1']['low_shelf']['cutoff_freq_hz']
        flat_params['l1_low_shelf_q_factor'] = config['L1']['low_shelf']['q_factor']
        
        for i, band in enumerate(config['L1']['bands']):
            flat_params[f'l1_band{i}_gain_db'] = band['gain_db']
            flat_params[f'l1_band{i}_cutoff_freq'] = band['cutoff_freq_hz']
            flat_params[f'l1_band{i}_q_factor'] = band['q_factor']
        
        flat_params['l1_high_shelf_gain_db'] = config['L1']['high_shelf']['gain_db']
        flat_params['l1_high_shelf_cutoff_freq'] = config['L1']['high_shelf']['cutoff_freq_hz']
        flat_params['l1_high_shelf_q_factor'] = config['L1']['high_shelf']['q_factor']
        
        # Process L2 parameters
        flat_params['l2_low_shelf_gain_db'] = config['L2']['low_shelf']['gain_db']
        flat_params['l2_low_shelf_cutoff_freq'] = config['L2']['low_shelf']['cutoff_freq_hz']
        flat_params['l2_low_shelf_q_factor'] = config['L2']['low_shelf']['q_factor']
        
        for i, band in enumerate(config['L2']['bands']):
            flat_params[f'l2_band{i}_gain_db'] = band['gain_db']
            flat_params[f'l2_band{i}_cutoff_freq'] = band['cutoff_freq_hz']
            flat_params[f'l2_band{i}_q_factor'] = band['q_factor']
        
        flat_params['l2_high_shelf_gain_db'] = config['L2']['high_shelf']['gain_db']
        flat_params['l2_high_shelf_cutoff_freq'] = config['L2']['high_shelf']['cutoff_freq_hz']
        flat_params['l2_high_shelf_q_factor'] = config['L2']['high_shelf']['q_factor']
        
        return config, flat_params
    
    except FileNotFoundError:
        print(f"Error: File {yaml_file_path} not found.")
        return None, None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    yaml_file_path = 'eq_config.yaml'
    
    # Load parameters
    nested_config, flat_params = load_eq_parameters(yaml_file_path)
    
    if flat_params:
        # Print nested config for reference
        print("Nested Config:")
        print(nested_config)
        
        # Print flattened parameters
        print("\nFlattened Parameters:")
        for key, value in flat_params.items():
            print(f"{key}: {value}")
        
        # Pass flattened parameters to the function
        result = define_eq_filters(**flat_params)
        
        # Print the result (for demonstration)
        print("\nFunction Output:")
        print(result)