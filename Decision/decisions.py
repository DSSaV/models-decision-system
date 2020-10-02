
def decision(values, settings):
    """Suplementary function to log_return_calculation
    
    Parameters:
        array: values from a dataframe column
        settings: lower_threshold and upper_threshold values in a dictionary
    Returns:
        array: an array holding buy/sell/hold decisions
    """
    # ARRAY FOR DECISIONS
    data = []
    
    # STARTING WITH NO SHARES HELD
    holding_shares = False
    
    # UPPER AND LOWER QUANTILES
    lower_threshold = values.quantile(settings["lower_threshold"])
    upper_threshold = values.quantile(settings["upper_threshold"])
    
    # MAKE A BUY/SELL/HOLD DECISION FOR EACH VALUE
    for value in values:
        print(value)
        if ((value < lower_threshold) & (holding_shares == True)):
            data.append("sell")
            holding_shares = False
        elif value > upper_threshold:
            data.append("buy")
            holding_shares = True
        else:
            data.append("hold")
            
    return data

def log_return_calculation(dataframe, settings):
    """
    Calculates the log return for every column in a dataframe.
    Log return = Log(Pt/Pt-1)
    
    Parameters:
        dataframe: pd.dataframe with columns containing values
        settings: dictionary cotaining lower_threshold and upper_threshold as values between 0 and 1
    Returns:
        dataframe: pd.dataframe which contains the buy/sell/hold decisions calculated from the input dataframe
    """
    # CREATE A DATAFRAME THAT HOLDS THE LOG RETURN VALUES
    dataframe_log = pd.DataFrame()
    
    # LOOP THROUGH EACH COLUMN IN THE INPUT DATAFRAME
    for index, column in enumerate(dataframe.columns):
        
        # APPEND _LOG TO THE COLUMN NAME 
        name = column + "_log"
        
        # CALCULATE LOG RETURN VALUE WITH CURRENT AND PREVIOUS
        dataframe_log[name] = decision(np.log(dataframe[column] / dataframe[column].shift(1)), settings)
    
    # REMOVE EMPTY VALUES AND RETURN THE LOG DATAFRAME
    return dataframe_log.dropna()