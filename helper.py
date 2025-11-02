from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = [url for message in df['message'] for url in extract.find_urls(message)]
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts()
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'}
    )
    return x, percent_df

def create_wordcloud(selected_user, df):
    with open('stop_english.txt', 'r') as f:
        stop_words = f.read()
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    temp['message'] = temp['message'].apply(lambda msg: " ".join([word for word in msg.lower().split() if word not in stop_words]))
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(temp['message'].str.cat(sep=" "))

def most_common_words(selected_user, df):
    with open('stop_english.txt', 'r') as f:
        stop_words = f.read()
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]
    words = [word for message in temp['message'] for word in message.lower().split() if word not in stop_words]
    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])

def emoji_helper(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    all_emojis = []
    for message in df['message']:
        # Extract emojis using emoji.emoji_list()
        extracted = emoji.emoji_list(message)
        if extracted:
            all_emojis.extend([e['emoji'] for e in extracted])
        else:
            # Fallback: character-level detection
            all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_counts = Counter(all_emojis)
    return pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values(by='count', ascending=False)

def emoji_emotion_summary(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]

    emotion_map = {
        'happy': ['😀', '😃', '😄', '😁', '😊', '😆', '😅', '🤣', '😂', '😸'],
        'sad': ['😢', '😭', '😞', '😔', '😟', '😩', '😫', '😿'],
        'angry': ['😠', '😡', '🤬', '👿'],
        'annoyed': ['😒', '🙄', '😤', '😑'],
    }

    all_emojis = []
    for message in df['message']:
        try:
            extracted = emoji.emoji_list(message)
            all_emojis.extend([e['emoji'] for e in extracted])
        except:
            all_emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emotion_counts = {}
    for emotion, symbols in emotion_map.items():
        emotion_counts[emotion] = sum(all_emojis.count(e) for e in symbols)

    return pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count'])

def sentiment_analysis(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    sid = SentimentIntensityAnalyzer()
    sentiments = df['message'].apply(lambda msg: sid.polarity_scores(msg)['compound'])
    sentiment_df = pd.DataFrame({
        'message': df['message'],
        'sentiment_score': sentiments
    })
    sentiment_df['sentiment'] = sentiment_df['sentiment_score'].apply(
        lambda score: 'positive' if score > 0.2 else 'negative' if score < -0.2 else 'neutral'
    )
    summary_df = sentiment_df['sentiment'].value_counts().reset_index()
    summary_df.columns = ['sentiment', 'messages']
    return summary_df

def monthly_time_line(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    time_line = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time_line['time'] = time_line.apply(lambda row: f"{row['month']} - {row['year']}", axis=1)
    return time_line

def daily_timeline(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()