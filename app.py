import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8", errors="ignore")
    df = preprocessor.preprocess(data)

    df['only_date'] = pd.to_datetime(df['only_date'])

    st.sidebar.subheader("Filter by Date Range")
    start_date = st.sidebar.date_input("Start Date", df['only_date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['only_date'].max().date())
    df = df[(df['only_date'].dt.date >= start_date) & (df['only_date'].dt.date <= end_date)]

    st.markdown(f"**Analyzing messages from {start_date} to {end_date}**")

    # ✅ Show chat messages at the top
    st.title("📄 Chat Preview")
    st.dataframe(df[['dates', 'user', 'message']], use_container_width=False)

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Top statistics
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        time_line = helper.monthly_time_line(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(time_line['time'], time_line['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            st.pyplot(fig)

        # Busiest user
        if selected_user == 'overall':
            st.title("Most Busy User")
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df, use_container_width=False)

        # Word cloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Most common words
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df['word'], most_common_df['count'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ✅ Emoji analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        st.subheader("All Emojis Used in Chat")
        st.dataframe(emoji_df, use_container_width=False)

       

        # ✅ Emoji emotion summary
        st.title("Emoji Emotion Summary")
        emotion_df = helper.emoji_emotion_summary(selected_user, df)
        st.dataframe(emotion_df, use_container_width=False)

        fig, ax = plt.subplots()
        ax.bar(emotion_df['emotion'], emotion_df['count'], color=['green', 'blue', 'red', 'orange'])
        st.pyplot(fig)

        # ✅ Sentiment analysis
        st.title("Sentiment Analysis")
        sentiment_df = helper.sentiment_analysis(selected_user, df)
        st.dataframe(sentiment_df, use_container_width=False)

        fig, ax = plt.subplots()
        ax.pie(sentiment_df['messages'], labels=sentiment_df['sentiment'], autopct="%0.2f")
        st.pyplot(fig)
