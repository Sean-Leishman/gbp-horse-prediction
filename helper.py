
from datetime import date

month_dict = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10',
              'Nov': '11', 'Dec': '12'}


def convertStringIntoDate(newDate):
    # print(newDate)
    newDate = [x.strip() for x in newDate.split(" ")]
    newDate[0] = int(newDate[0])
    newDate[1] = int(month_dict[newDate[1]])
    newDate[2] = int(newDate[2])
    return date(newDate[2], newDate[1], newDate[0])


def convertDateToInt(currDate, minDate):
    return (currDate - minDate).days


type_dict = {
    0: 'hurdle_win_percent',
    1: 'flat_win_percent',
    2: 'chase_win_percent'
}
going_dict = {
    0: 'heavy_win_percent',
    1: 'heavy_win_percent',
    2: 'heavy_win_percent',
    3: 'heavy_win_percent',
    4: 'good_soft_win_percent',
    5: 'good_soft_win_percent',
    6: 'good_soft_win_percent',
    7: 'good_win_percent',
    8: 'good_win_percent',
    9: 'good_firm_win_percent',
    10: 'firm_win_percent',
    11: 'firm_win_percent',
}


dist_dict = {
    4: '0_6_win_percent',
    5: '0_6_win_percent',
    6: '0_6_win_percent',
    7: '7_9_win_percent',
    8: '7_9_win_percent',
    9: '7_9_win_percent',
    10: '10_13_win_percent',
    11: '10_13_win_percent',
    12: '10_13_win_percent',
    13: '14_20_win_percent',
    14: '14_20_win_percent',
    15: '14_20_win_percent',
    16: '14_20_win_percent',
    17: '14_20_win_percent',
    18: '14_20_win_percent',
    19: '14_20_win_percent',
    20: '14_20_win_percent',
    21: '21_30_win_percent',
    22: '21_30_win_percent',
    23: '21_30_win_percent',
    24: '21_30_win_percent',
    25: '21_30_win_percent',
    26: '21_30_win_percent',
    27: '21_30_win_percent',
    28: '21_30_win_percent',
    29: '21_30_win_percent',
    30: '21_30_win_percent',
    31: '31_40_win_percent',
    32: '31_40_win_percent',
    33: '31_40_win_percent',
    34: '31_40_win_percent',
    35: '31_40_win_percent',
    36: '31_40_win_percent',
    37: '31_40_win_percent',
    38: '31_40_win_percent',
    39: '31_40_win_percent',
    40: '31_40_win_percent'
}

race_class_to_scale_dict = {
    '(Class 7)': 1,
    '(Class 6)': 2,
    '(Class 5)': 3,
    '(Class 4)': 4,
    '(Class 3)': 5,
    '(Class 2)': 6,
    '(Class 1)': 7,
}

going_to_scale_dict = {
    'Muddy': 0,
    'Sloppy': 0,
    'Very Soft': 1,
    'Heavy': 1,
    'Soft To Heavy': 2,
    'Soft': 3,
    'Yielding To Soft': 4,
    'Good To Soft': 5,
    'Yielding': 6,
    'Good To Yielding': 7,
    'Good':8,
    'Good To Firm':9,
    'Firm': 10,
    'Frozen': 11,

    'Slow': 3,
    'Sta   ndard To Slow': 5,
    'Standard To Slow': 5,
    'Standard': 7,
    'Fast': 10, 
    'Abandoned': 0,
}

going_original_name_dict = {
    'Muddy': 'heavy',
    'Sloppy': 'heavy',
    'Very Soft': 'heavy',
    'Heavy': 'heavy',
    'Soft To Heavy': 'soft',
    'Soft': 'soft',
    'Yielding To Soft': 'good_soft',
    'Good To Soft': 'good_soft',
    'Yielding': 'good_soft',
    'Good To Yielding': 'good',
    'Good':'good',
    'Good To Firm':'good_firm',
    'Firm': 'firm',
    'Frozen': 'firm',

    'Slow': 'soft',
    'Sta   ndard To Slow': 'good_soft',
    'Standard To Slow': 'good_soft',
    'Standard': 'good',
    'Fast': 'firm', 
    'Abandoned': 'firm',
}