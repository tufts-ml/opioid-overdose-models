'''
Makes the timestep column in cook_county_total_script.py
'''

def map_season_semi(semiannual):
    return 'jan-jun' if semiannual == '1' else 'jul-dec'

def map_season_q(q):
    if q == 1:
        return 'jan-mar'
    elif q == 2:
        return 'apr-jun'
    elif q == 3:
        return 'jul-sep'
    elif q == 4:
        return 'oct-dec'
    
# Create month column
def map_month(month):
    if month == 1:
        return 'jan'
    elif month == 2:
        return 'feb'
    elif month == 3:
        return 'mar'
    elif month == 4:
        return 'apr'
    elif month == 5:
        return 'may'
    elif month == 6:
        return 'jun'
    elif month == 7:
        return 'jul'
    elif month == 8:
        return 'aug'
    elif month == 9:
        return 'sep'
    elif month == 10:
        return 'oct'
    elif month == 11:
        return 'nov'
    elif month == 12:
        return 'dec'

def map_biweek(biweek):
    if biweek == 1:
        return 'bw01'
    elif biweek == 2:
        return 'bw02'
    elif biweek == 3:
        return 'bw03'
    elif biweek == 4:
        return 'bw04'
    elif biweek == 5:
        return 'bw05'
    elif biweek == 6:
        return 'bw06'
    elif biweek == 7:
        return 'bw07'
    elif biweek == 8:
        return 'bw08'
    elif biweek == 9:
        return 'bw09'
    elif biweek == 10:
        return 'bw10'
    elif biweek == 11:
        return 'bw11'
    elif biweek == 12:
        return 'bw12'
    elif biweek == 13:
        return 'bw13'
    elif biweek == 14:
        return 'bw14'
    elif biweek == 15:
        return 'bw15'
    elif biweek == 16:
        return 'bw16'
    elif biweek == 17:
        return 'bw17'
    elif biweek == 18:
        return 'bw18'
    elif biweek == 19:
        return 'bw19'
    elif biweek == 20:
        return 'bw20'
    elif biweek == 21:
        return 'bw21'
    elif biweek == 22:
        return 'bw22'
    elif biweek == 23:
        return 'bw23'
    elif biweek == 24:
        return 'bw24'
    elif biweek == 25:
        return 'bw25'
    elif biweek == 26:
        return 'bw26'
    elif biweek == 27:
        return 'bw27'

def map_week(week):
    if int(week) < 10:
        return 'week0' + str(week)
    else:
        return 'week' + str(week)