import re
import pandas as pd

def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    except:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

    df.rename(columns={'message_date': 'dates'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['dates'].dt.date
    df['year'] = df['dates'].dt.year
    df['month_num'] = df['dates'].dt.month
    df['month'] = df['dates'].dt.month_name()
    df['day'] = df['dates'].dt.day
    df['day_name'] = df['dates'].dt.day_name()
    df['hour'] = df['dates'].dt.hour
    df['minute'] = df['dates'].dt.minute

    return df