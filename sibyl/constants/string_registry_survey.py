INTERESTED_GROUPS = []

PREDEFINED_ENCODING_OPINIONQA = [
    'age_18-29', 'age_30-49', 'age_50-64', 'age_65+',
    'race_white', 'race_black', 'race_hispanic', 'race_asian', 'race_other',
    'sex_male', 'sex_female', 'sex_other',
    'education_less than high school', 'education_high school graduate', 'education_some college, no degree', "education_associate's degree", 'education_college graduate/some postgrad', 'education_postgraduate',
    'income_less than $30,000', 'income_$30,000-$50,000', 'income_$50,000-$75,000', 'income_$75,000-$100,000', 'income_$100,000 or more',
    'cregion_northeast', 'cregion_midwest', 'cregion_south', 'cregion_west',
    'relig_protestant', 'relig_roman catholic', 'relig_mormon', 'relig_orthodox', 'relig_jewish', 'relig_muslim', 'relig_buddhist', 'relig_hindu', 'relig_atheist', 'relig_agnostic', 'relig_other', 'relig_nothing in particular',
    'polparty_republican', 'polparty_democrat', 'polparty_independent', 'polparty_something else',
    'polideology_very conservative', 'polideology_conservative', 'polideology_moderate', 'polideology_liberal', 'polideology_very liberal'
]

PREDEFINED_ENCODING_DUNNING_KRUGER = [
    "pre_accuracy_pre_correct_0", "pre_accuracy_pre_correct_1", "pre_accuracy_pre_correct_2", "pre_accuracy_pre_correct_3", "pre_accuracy_pre_correct_4", "pre_accuracy_pre_correct_5", "pre_accuracy_pre_correct_6", "pre_accuracy_pre_correct_7", "pre_accuracy_pre_correct_8", "pre_accuracy_pre_correct_9", "pre_accuracy_pre_correct_10", "pre_accuracy_pre_correct_11", "pre_accuracy_pre_correct_12", "pre_accuracy_pre_correct_13", "pre_accuracy_pre_correct_14", "pre_accuracy_pre_correct_15", "pre_accuracy_pre_correct_16", "pre_accuracy_pre_correct_17", "pre_accuracy_pre_correct_18", "pre_accuracy_pre_correct_19", "pre_accuracy_pre_correct_20",
    "pre_percentile_pre_percentile_0-10", "pre_percentile_pre_percentile_11-20", "pre_percentile_pre_percentile_21-30", "pre_percentile_pre_percentile_31-40", "pre_percentile_pre_percentile_41-50", "pre_percentile_pre_percentile_51-60", "pre_percentile_pre_percentile_61-70", "pre_percentile_pre_percentile_71-80", "pre_percentile_pre_percentile_81-90", "pre_percentile_pre_percentile_91-100",
    "pre_average_difficulty_pre_average_difficulty_1", "pre_average_difficulty_pre_average_difficulty_2", "pre_average_difficulty_pre_average_difficulty_3", "pre_average_difficulty_pre_average_difficulty_4", "pre_average_difficulty_pre_average_difficulty_5", "pre_average_difficulty_pre_average_difficulty_6", "pre_average_difficulty_pre_average_difficulty_7", "pre_average_difficulty_pre_average_difficulty_8", "pre_average_difficulty_pre_average_difficulty_9", "pre_average_difficulty_pre_average_difficulty_10",
    "pre_self_difficulty_pre_self_difficulty_1", "pre_self_difficulty_pre_self_difficulty_2", "pre_self_difficulty_pre_self_difficulty_3", "pre_self_difficulty_pre_self_difficulty_4", "pre_self_difficulty_pre_self_difficulty_5", "pre_self_difficulty_pre_self_difficulty_6", "pre_self_difficulty_pre_self_difficulty_7", "pre_self_difficulty_pre_self_difficulty_8", "pre_self_difficulty_pre_self_difficulty_9", "pre_self_difficulty_pre_self_difficulty_10",
    "post_accuracy_post_correct_0", "post_accuracy_post_correct_1", "post_accuracy_post_correct_2", "post_accuracy_post_correct_3", "post_accuracy_post_correct_4", "post_accuracy_post_correct_5", "post_accuracy_post_correct_6", "post_accuracy_post_correct_7", "post_accuracy_post_correct_8", "post_accuracy_post_correct_9", "post_accuracy_post_correct_10", "post_accuracy_post_correct_11", "post_accuracy_post_correct_12", "post_accuracy_post_correct_13", "post_accuracy_post_correct_14", "post_accuracy_post_correct_15", "post_accuracy_post_correct_16", "post_accuracy_post_correct_17", "post_accuracy_post_correct_18", "post_accuracy_post_correct_19", "post_accuracy_post_correct_20",
    "post_percentile_post_percentile_0-10", "post_percentile_post_percentile_11-20", "post_percentile_post_percentile_21-30", "post_percentile_post_percentile_31-40", "post_percentile_post_percentile_41-50", "post_percentile_post_percentile_51-60", "post_percentile_post_percentile_61-70", "post_percentile_post_percentile_71-80", "post_percentile_post_percentile_81-90", "post_percentile_post_percentile_91-100",
    "post_average_difficulty_post_average_difficulty_1", "post_average_difficulty_post_average_difficulty_2", "post_average_difficulty_post_average_difficulty_3", "post_average_difficulty_post_average_difficulty_4", "post_average_difficulty_post_average_difficulty_5", "post_average_difficulty_post_average_difficulty_6", "post_average_difficulty_post_average_difficulty_7", "post_average_difficulty_post_average_difficulty_8", "post_average_difficulty_post_average_difficulty_9", "post_average_difficulty_post_average_difficulty_10",
    "post_self_difficulty_post_self_difficulty_1", "post_self_difficulty_post_self_difficulty_2", "post_self_difficulty_post_self_difficulty_3", "post_self_difficulty_post_self_difficulty_4", "post_self_difficulty_post_self_difficulty_5", "post_self_difficulty_post_self_difficulty_6", "post_self_difficulty_post_self_difficulty_7", "post_self_difficulty_post_self_difficulty_8", "post_self_difficulty_post_self_difficulty_9", "post_self_difficulty_post_self_difficulty_10",
]

PREDEFINED_ENCODING = {
    "ATP": PREDEFINED_ENCODING_OPINIONQA,
    "Twin": PREDEFINED_ENCODING_OPINIONQA,
    "dunning_kruger": PREDEFINED_ENCODING_DUNNING_KRUGER,
    "eedi": {},
}

DELIMITER = '_W'

INVALID_FLAGS = [
    '',
    'refused',
    'nan',
    'no answer',
    'did not receive question',
    'unclear',
    'dk',
    'something else, specify:',
]

VALID_TRAITS = [
    "age", "race", "sex", "education", "income",
    "cregion", "relig", "polparty", "polideology",
    "pre_accuracy", "pre_percentile", "pre_average_difficulty", "pre_self_difficulty",
    "post_accuracy", "post_percentile", "post_average_difficulty", "post_self_difficulty",
]

Twin_AGE_COLUMN = ['BLOCK0QID13_W1']
Twin_SEX_COLUMN = ['BLOCK0QID12_W1']
Twin_EDUCATION_COLUMN = ['BLOCK0QID14_W1']
Twin_CREGION_COLUMN = ['BLOCK0QID11_W1']
Twin_POLPARTY_COLUMN = ['BLOCK0QID20_W1']
Twin_POLIDEOLOGY_COLUMN = ['BLOCK0QID22_W1']
Twin_INCOME_COLUMN = ['BLOCK0QID21_W1']
Twin_RELIG_COLUMN = ['BLOCK0QID18_W1']
Twin_RACE_COLUMN = ['BLOCK0QID15_W1']

dunning_kruger_PRE_ACCURACY_COLUMN = ['pre_accuracy']
dunning_kruger_PRE_PERCENTILE_COLUMN = ['pre_percentile']
dunning_kruger_PRE_AVERAGE_DIFFICULTY_COLUMN = ['pre_average_difficulty']
dunning_kruger_PRE_SELF_DIFFICULTY_COLUMN = ['pre_self_difficulty']
dunning_kruger_POST_ACCURACY_COLUMN = ['post_accuracy']
dunning_kruger_POST_PERCENTILE_COLUMN = ['post_percentile']
dunning_kruger_POST_AVERAGE_DIFFICULTY_COLUMN = ['post_average_difficulty']
dunning_kruger_POST_SELF_DIFFICULTY_COLUMN = ['post_self_difficulty']

ATP_AGE_COLUMN = ['F_AGECAT_FINAL', 'F_AGECAT']
ATP_SEX_COLUMN = ['F_SEX_FINAL', 'F_SEX', 'F_GENDER']
ATP_EDUCATION_COLUMN = ['F_EDUCCAT2', 'F_EDUCCAT2_FINAL']
ATP_CREGION_COLUMN = ['F_CREGION', 'F_CREGION_FINAL']
ATP_POLPARTY_COLUMN = ['F_PARTY_FINAL']
ATP_POLIDEOLOGY_COLUMN = ['F_IDEO', 'F_IDEO_FINAL']
ATP_INCOME_COLUMN = ['F_INCOME', 'F_INC_SDT1', 'F_INCOME_FINAL', 'INC_SDT1_W54']
ATP_RELIG_COLUMN = ['F_RELIG', 'F_RELIG_FINAL']
ATP_RACE_COLUMN = ['F_RACETHNMOD', 'F_RACETHN_RECRUITMENT', 'F_RACETHN']

ATP_CONVERSION_TABLE = {
    # age
    'f_agecat_final' : {
        '18-29' : '18-29', 
        '30-49' : '30-49', 
        '50-64' : '50-64', 
        '65+' : '65+'
    },
    'f_agecat' : {
        '18-29' : '18-29',
        '30-49' : '30-49',
        '50-64' : '50-64',
        '65+' : '65+'
    },
    # sex or gender
    'f_sex_final' : {
        'male' : 'male',
        'female' : 'female'
    },
    'f_sex' : {
        'male' : 'male',
        'female' : 'female'
    },
    'f_gender' : {
        'a man' : 'male',
        'a woman' : 'female',
        'in some other way' : 'in some other way'
    },
    # education
    'f_educcat2' : {
        'less than high school' : 'less than high school',
        'high school graduate' : 'high school graduate',
        'some college, no degree' : 'some college, no degree',
        "associate's degree" : "associate's degree",
        "associate’s degree" : "associate's degree",
        'college graduate/some post grad' : 'college graduate/some postgrad',
        'college graduate/some postgrad' : 'college graduate/some postgrad',
        'postgraduate' : 'postgraduate'
    },
    'f_educcat2_final' : {
        'less than high school' : 'less than high school',
        'high school graduate' : 'high school graduate',
        'some college, no degree' : 'some college, no degree',
        "associate's degree" : "associate's degree",
        "associate’s degree" : "associate's degree",
        'college graduate/some post grad' : 'college graduate/some postgrad',
        'college graduate/some postgrad' : 'college graduate/some postgrad',
        'postgraduate' : 'postgraduate'
    },
    # region
    'f_cregion' : {
        'northeast' : 'northeast',
        'midwest' : 'midwest',
        'south' : 'south',
        'west' : 'west'
    },
    'f_cregion_final' : {
        'northeast' : 'northeast',
        'midwest' : 'midwest',
        'south' : 'south',
        'west' : 'west'
    },
    # political affiliation
    'f_party_final'  : {
        'republican' : 'republican', 
        'democrat' : 'democrat',
        'independent' : 'independent',
        'something else' : 'something else'
    },
    # political ideology
    'f_ideo' : {
        'very conservative' : 'very conservative',
        'conservative' : 'conservative',
        'moderate' : 'moderate',
        'liberal' : 'liberal',
        'very liberal' : 'very liberal'
    },
    'f_ideo_final' : {
        'very conservative' : 'very conservative',
        'conservative' : 'conservative',
        'moderate' : 'moderate',
        'liberal' : 'liberal',
        'very liberal' : 'very liberal'
    },
    # income
    'f_income': {
        'less than $10,000' : 'less than $30,000',
        '$10,000 to less than $20,000' : 'less than $30,000',
        '$20,000 to less than $30,000' : 'less than $30,000',
        '$30,000 to less than $40,000' : '$30,000-$50,000',
        '$40,000 to less than $50,000' : '$30,000-$50,000',
        '$50,000 to less than $75,000' : '$50,000-$75,000',
        '$75,000 to less than $100,000' : '$75,000-$100,000',
        '$100,000 to less than $150,000' : '$100,000 or more',
        '$150,000 or more' : '$100,000 or more'
    },
    'f_inc_sdt1': {
        'less than $30,000': 'less than $30,000',
        '$30,000 to less than $40,000': '$30,000-$50,000',
        '$40,000 to less than $50,000': '$30,000-$50,000',
        '$50,000 to less than $60,000': '$50,000-$75,000',
        '$60,000 to less than $70,000': '$50,000-$75,000',
        '$70,000 to less than $80,000': '$50,000-$75,000',
        '$80,000 to less than $90,000': '$75,000-$100,000',
        '$90,000 to less than $100,000': '$75,000-$100,000',
        '$100,000 or more': '$100,000 or more'
    },
    'f_income_final': {
        'less than $10,000' : 'less than $30,000',
        '10 to under $20,000' : 'less than $30,000',
        '20 to under $30,000' : 'less than $30,000',
        '30 to under $40,000' : '$30,000-$50,000',
        '40 to under $50,000' : '$30,000-$50,000',
        '50 to under $75,000' : '$50,000-$75,000',
        '75 to under $100,000' : '$75,000-$100,000',
        '100 to under $150,000 [or]' : '$100,000 or more',
        '$150,000 or more' : '$100,000 or more'
    },
    'inc_sdt1_w54': {
        'less than $30,000': 'less than $30,000',
        '$30,000 to less than $40,000': '$30,000-$50,000',
        '$40,000 to less than $50,000': '$30,000-$50,000',
        '$50,000 to less than $60,000': '$50,000-$75,000',
        '$60,000 to less than $70,000': '$50,000-$75,000',
        '$70,000 to less than $80,000': '$50,000-$75,000',
        '$80,000 to less than $90,000': '$75,000-$100,000',
        '$90,000 to less than $100,000': '$75,000-$100,000',
        '$100,000 or more': '$100,000 or more',
    },
    # religion
    'f_relig' : {
        'protestant' : 'protestant', 
        'roman catholic' : 'roman catholic',
        'mormon' : 'mormon',
        'orthodox' : 'orthodox',
        'jewish' : 'jewish',
        'muslim' : 'muslim',
        'buddhist' : 'buddhist',
        'hindu' : 'hindu',
        'atheist' : 'atheist',
        'agnostic' : 'agnostic',
        'other' : 'other',
        'nothing in particular' : 'nothing in particular',
        'something else, specify' : 'other',
    },
    'f_relig_final' : {
        'protestant' : 'protestant', 
        'roman catholic' : 'roman catholic',
        'mormon' : 'mormon',
        'orthodox' : 'orthodox',
        'jewish' : 'jewish',
        'muslim' : 'muslim',
        'buddhist' : 'buddhist',
        'hindu' : 'hindu',
        'atheist' : 'atheist',
        'agnostic' : 'agnostic',
        'other' : 'other',
        'nothing in particular' : 'nothing in particular',
        'something else, specify' : 'other',
    },
    # race or ethnicity
    'f_racethnmod' : {
        'white non-hispanic' : 'white',
        'white' : 'white',
        'black non-hispanic' : 'black',
        'black' : 'black',
        'hispanic' : 'hispanic',
        'asian non-hispanic' : 'asian',
        'asian' : 'asian',
        'other' : 'other'
    },
    'f_racethn_recruitment' : {
        'white non-hispanic' : 'white',
        'black non-hispanic' : 'black',
        'hispanic' : 'hispanic',
        'other' : 'other',
    },
    'f_racethn' : {
        'white non-hispanic' : 'white',
        'black non-hispanic' : 'black',
        'hispanic' : 'hispanic',
        'other' : 'other'
    }
}

INDIV_FEAT_ENCODING_OPINIONQA = {
    'age': {
        '18-29': 0,
        '30-49': 1,
        '50-64': 2,
        '65+': 3
    },
    'race': {
        'white': 0,
        'black': 1,
        'hispanic': 2,
        'asian': 3,
        'other': 4
    },
    'sex': {
        'male': 0,
        'female': 1,
        'other': 2
    },
    'education': {
        'less than high school': 0,
        'high school graduate': 1,
        'some college, no degree': 2,
        "associate's degree": 3,
        'college graduate/some postgrad': 4,
        'postgraduate': 5
    },
    'income': {
        'less than $30,000': 0,
        '$30,000-$50,000': 1,
        '$50,000-$75,000': 2,
        '$75,000-$100,000': 3,
        '$100,000 or more': 4
    },
    'cregion': {
        'northeast': 0,
        'midwest': 1,
        'south': 2,
        'west': 3
    },
    'relig': {
        'protestant': 0,
        'roman catholic': 1,
        'mormon': 2,
        'orthodox': 3,
        'jewish': 4,
        'muslim': 5,
        'buddhist': 6,
        'hindu': 7,
        'atheist': 8,
        'agnostic': 9,
        'other': 10,
        'nothing in particular': 11
    },
    'polparty': {
        'republican': 0,
        'democrat': 1,
        'independent': 2,
        'something else': 3
    },
    'polideology': {
        'very conservative': 0,
        'conservative': 1,
        'moderate': 2,
        'liberal': 3,
        'very liberal': 4
    }
}

INDIV_FEAT_ENCODING_DUNNING_KRUGER = {
    'pre_accuracy': {
        'pre_correct_0': 0,
        'pre_correct_1': 1,
        'pre_correct_2': 2,
        'pre_correct_3': 3,
        'pre_correct_4': 4,
        'pre_correct_5': 5,
        'pre_correct_6': 6,
        'pre_correct_7': 7,
        'pre_correct_8': 8,
        'pre_correct_9': 9,
        'pre_correct_10': 10,
        'pre_correct_11': 11,
        'pre_correct_12': 12,
        'pre_correct_13': 13,
        'pre_correct_14': 14,
        'pre_correct_15': 15,
        'pre_correct_16': 16,
        'pre_correct_17': 17,
        'pre_correct_18': 18,
        'pre_correct_19': 19,
        'pre_correct_20': 20
    },
    'pre_percentile': {
        'pre_percentile_0-10': 0,
        'pre_percentile_11-20': 1,
        'pre_percentile_21-30': 2,
        'pre_percentile_31-40': 3,
        'pre_percentile_41-50': 4,
        'pre_percentile_51-60': 5,
        'pre_percentile_61-70': 6,
        'pre_percentile_71-80': 7,
        'pre_percentile_81-90': 8,
        'pre_percentile_91-100': 9
    },
    'pre_average_difficulty': {
        'pre_average_difficulty_1': 0,
        'pre_average_difficulty_2': 1,
        'pre_average_difficulty_3': 2,
        'pre_average_difficulty_4': 3,
        'pre_average_difficulty_5': 4,
        'pre_average_difficulty_6': 5,
        'pre_average_difficulty_7': 6,
        'pre_average_difficulty_8': 7,
        'pre_average_difficulty_9': 8,
        'pre_average_difficulty_10': 9
    },
    'pre_self_difficulty': {
        'pre_self_difficulty_1': 0,
        'pre_self_difficulty_2': 1,
        'pre_self_difficulty_3': 2,
        'pre_self_difficulty_4': 3,
        'pre_self_difficulty_5': 4,
        'pre_self_difficulty_6': 5,
        'pre_self_difficulty_7': 6,
        'pre_self_difficulty_8': 7,
        'pre_self_difficulty_9': 8,
        'pre_self_difficulty_10': 9
    },
    'post_accuracy': {
        'post_correct_0': 0,
        'post_correct_1': 1,
        'post_correct_2': 2,
        'post_correct_3': 3,
        'post_correct_4': 4,
        'post_correct_5': 5,
        'post_correct_6': 6,
        'post_correct_7': 7,
        'post_correct_8': 8,
        'post_correct_9': 9,
        'post_correct_10': 10,
        'post_correct_11': 11,
        'post_correct_12': 12,
        'post_correct_13': 13,
        'post_correct_14': 14,
        'post_correct_15': 15,
        'post_correct_16': 16,
        'post_correct_17': 17,
        'post_correct_18': 18,
        'post_correct_19': 19,
        'post_correct_20': 20
    },
    'post_percentile': {
        'post_percentile_0-10': 0,
        'post_percentile_11-20': 1,
        'post_percentile_21-30': 2,
        'post_percentile_31-40': 3,
        'post_percentile_41-50': 4,
        'post_percentile_51-60': 5,
        'post_percentile_61-70': 6,
        'post_percentile_71-80': 7,
        'post_percentile_81-90': 8,
        'post_percentile_91-100': 9
    },
    'post_average_difficulty': {
        'post_average_difficulty_1': 0,
        'post_average_difficulty_2': 1,
        'post_average_difficulty_3': 2,
        'post_average_difficulty_4': 3,
        'post_average_difficulty_5': 4,
        'post_average_difficulty_6': 5,
        'post_average_difficulty_7': 6,
        'post_average_difficulty_8': 7,
        'post_average_difficulty_9': 8,
        'post_average_difficulty_10': 9
    },
    'post_self_difficulty': {
        'post_self_difficulty_1': 0,
        'post_self_difficulty_2': 1,
        'post_self_difficulty_3': 2,
        'post_self_difficulty_4': 3,
        'post_self_difficulty_5': 4,
        'post_self_difficulty_6': 5,
        'post_self_difficulty_7': 6,
        'post_self_difficulty_8': 7,
        'post_self_difficulty_9': 8,
        'post_self_difficulty_10': 9
    }
}

INDIV_FEAT_ENCODING = {
    "ATP": INDIV_FEAT_ENCODING_OPINIONQA,
    "Twin": INDIV_FEAT_ENCODING_OPINIONQA,
    "dunning_kruger": INDIV_FEAT_ENCODING_DUNNING_KRUGER,
    "eedi": {},
}