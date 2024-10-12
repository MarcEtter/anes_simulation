def code_to_category(code,category):
    try:
        return category[code]
    except:
        return 'no matching code'

race_category = {
    1: 'White non-Hispanic',
    2: 'Black non-Hispanic',
    3: 'Asian or Pacific Islander, non-Hispanic',
    4: 'American Indian or Alaska Native non-Hispanic',
    5: 'Hispanic',
    6: 'Other or multiple races, non-Hispanic',
    7: 'Non-white and non-black'
}

educ_category = {
    1: '8 grades or less',
    2: '9-12 grades',
    3: '12 grades',
    4: '12 grades plus non-academic training',
    5: 'Some college, no degree; AA degree',
    6: 'BA level degree',
    7: 'Advanced degree'}

state_name = {
        1: 'Alabama',
        2:   'Alaska',
         4:  'Arizona',
         5:   'Arkansas',
         6:   'California',
         8:   'Colorado',
         9:   'Connecticut',
        10:   'Delaware',
        11:   'Washington DC',
        12:   'Florida',
        13:   'Georgia',
        15:   'Hawaii',
        16:   'Idaho',
        17:   'Illinois',
        18:   'Indiana',
        19:   'Iowa',
        20:   'Kansas',
        21:   'Kentucky',
        22:   'Louisiana',
        23:   'Maine',
        24:   'Maryland',
        25:   'Massachusetts',
        26:   'Michigan',
        27:   'Minnesota',
        28:   'Mississippi',
        29:   'Missouri',
        30:   'Montana',
        31:   'Nebraska',
        32:   'Nevada',
        33:   'New Hampshire',
        34:   'New Jersey',
        35:   'New Mexico',
        36:   'New York',
        37:   'North Carolina',
        38:   'North Dakota',
        39:   'Ohio',
        40:   'Oklahoma',
        41:   'Oregon',
        42:   'Pennsylvania',
        44:   'Rhode Island',
        45:   'South Carolina',
        46:   'South Dakota',
        47:   'Tennessee',
        48:   'Texas',
        49:   'Utah',
        50:   'Vermont',
        51:   'Virginia',
        53:   'Washington',
        54:   'West Virginia',
        55:   'Wisconsin',
        56:   'Wyoming'}

code_to_fips = {
        'Alabama':1,
        'Alaska':2,
         'Arizona':4,
         'Arkansas':5,
         'California':6,
         'Colorado':8,
         'Connecticut':9,
         'Delaware':10,
         'Washington DC':11,
         'Florida':12,
         'Georgia':13,
         'Hawaii':15,
         'Idaho':16,
         'Illinois':17,
         'Indiana':18,
         'Iowa':19,
         'Kansas':20,
         'Kentucky':21,
         'Louisiana':22,
         'Maine':23,
         'Maryland':24,
         'Massachusetts':25,
         'Michigan':26,
         'Minnesota':27,
         'Mississippi':28,
         'Missouri':29,
         'Montana':30,
         'Nebraska':31,
         'Nevada':32,
         'New Hampshire':33,
         'New Jersey':34,
         'New Mexico':35,
         'New York':36,
         'North Carolina':37,
         'North Dakota':38,
         'Ohio':39,
         'Oklahoma':40,
         'Oregon':41,
         'Pennsylvania':42,
         'Rhode Island':44,
         'South Carolina':45,
         'South Dakota':46,
         'Tennessee':47,
         'Texas':48,
         'Utah':49,
         'Vermont':50,
         'Virginia':51,
         'Washington':53,
         'West Virginia':54,
         'Wisconsin':55,
         'Wyoming':56,
         
         'AL':1,
         'AK':2,
         'AZ':4,
         'AR':5,
         'CA':6,
         'CO':8,
         'CT':9,
         'DE':10,
         'DC':11,
         'FL':12,
         'GA':13,
         'HI':15,
         'ID':16,
         'IL':17,
         'IN':18,
         'IA':19,
         'KS':20,
         'KY':21,
         'LA':22,
         'ME':23,
         'MD':24,
         'MA':25,
         'MI':26,
         'MN':27,
         'MS':28,
         'MO':29,
         'MT':30,
         'NE':31,
         'NV':32,
         'NH':33,
         'NJ':34,
         'NM':35,
         'NY':36,
         'NC':37,
         'ND':38,
         'OH':39,
         'OK':40,
         'OR':41,
         'PA':42,
         'RI':44,
         'SC':45,
         'SD':46,
         'TN':47,
         'TX':48,
         'UT':49,
         'VT':50,
         'VA':51,
         'WA':53,
         'WV':54,
         'WI':55,
         'WY':56
         }

incumbent = {
    1948: 0,
    1952: 0,
    1956: 1, 
    1960: 1, 
    1964: 0,
    1968: 0, 
    1972: 1,
    1976: 1, 
    1980: 0,
    1984: 1, 
    1988: 1, 
    1992: 1, 
    1996: 0,
    2000: 0,
    2004: 1,
    2008: 1,
    2012: 0,
    2016: 0,
    2020: 1,
    2024: 0
}

#note: this coding only applies to the use of the incumbent dictionary above,
#not to the coding of Democratic and Republican vote intention in the dataset
