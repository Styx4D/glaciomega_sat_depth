import torch
import numpy as np
import math
from datetime import datetime

def timestamp_to_delay( all_sent_time, encoder_type ):
    t0 = all_sent_time[0]
    t_last = all_sent_time[-1]
    t_diff = max( 1, (t_last-t0).days )
    if encoder_type == 'gla3':
        return torch.tensor( [(t-t0).days for t in all_sent_time ] )
    return torch.tensor( [(t-t0).days / t_diff for t in all_sent_time ] )

def days_from_winter_solstice(timestamps):
    result = []
    
    for ts in timestamps:
        year = ts.year
        
        # Winter solstice date (Dec 21)
        solstice = datetime(year, 12, 21)
        
        # If date is before Dec 21, solstice belongs to previous year winter season
        if ts < solstice:
            solstice = datetime(year - 1, 12, 21)
        
        delta_days = (ts - solstice).days
        result.append(delta_days)
    
    return torch.tensor(result)

