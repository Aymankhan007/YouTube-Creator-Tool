import itertools
import math
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build


# -------------------- CONSTANTS --------------------

STOPWORDS = {
    "a","about","after","all","also","an","and","are","as","at","be","because","been",
    "but","by","can","could","for","from","get","how","if","in","into","is","it","its",
    "just","like","make","new","not","of","on","or","our","out","over","the","this",
    "to","today","top","up","what","when","why","with","you","your",
}

TOPIC_TEMPLATES = [
    "Beginner's guide to {kw1}",
    "{kw1} vs {kw2}: which should you choose?",
    "How to master {kw1} in 30 days",
    "The truth about {kw1} and {kw2}",
    "{kw1} mistakes creators keep making",
    "{kw1} trends to watch this year",
    "Step-by-step {kw1} workflow",
    "{kw1} tools that save hours",
]


# -------------------- YOUTUBE HELPERS --------------------

def build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def extract_video_id(video_url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube video URL.")


def get_channel_id_from_video(client, video_url: str) -> str:
    video_id = extract_video_id(video_url)
    response = client.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("No video found for this URL.")
    return items[0]["snippet"]["channelId"]


def get_uploads_playlist_id(client, channel_id: str) -> str:
    response = client.channels().list(
        part="contentDetails", id=channel_id
    ).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError("Channel not found.")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def fetch_playlist_video_ids(client, playlist_id: str, max_videos: int) -> List[str]:
    video_ids = []
    next_page = None

    while len(video_ids) < max_videos:
        response = client.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=next_page,
        ).execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return video_ids


def fetch_video_details(client, video_ids: List[str]) -> List[Dict]:
    details = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        response = client.videos().list(
            part="snippet,statistics", id=",".join(chunk)
        ).execute()
        details.extend(response.get("items", []))
    return details


# -------------------- DATA PROCESSING --------------------

def build_dataframe(details: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in details:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        rows.append({
            "title": snippet.get("title"),
            "published_at": snippet.get("publishedAt"),
            "tags": ", ".join(snippet.get("tags", [])),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0)),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        df.sort_values("published_at", inplace=True)

    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return dict(avg_views=0, avg_likes=0, avg_comments=0, like_comment_ratio=0)

    avg_views = df["views"].mean()
    avg_likes = df["likes"].mean()
    avg_comments = df["comments"].mean()
    ratio = avg_likes / avg_comments if avg_comments else math.nan

    return {
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "like_comment_ratio": ratio,
    }


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def weighted_keyword_scores(df: pd.DataFrame) -> Counter:
    now = pd.Timestamp.utcnow()
    scores = Counter()

    for _, row in df.iterrows():
        months_since = (now - row["published_at"]).days / 30
        weight = 1 / (1 + months_since)

        tokens = tokenize(row["title"])
        tokens += tokenize(row["tags"])

        for token in tokens:
            scores[token] += weight

    return scores


def build_topic_predictions(df: pd.DataFrame, max_topics: int = 8):
    if df.empty:
        return [], []

    scores = weighted_keyword_scores(df)
    top_keywords = scores.most_common(8)
    keywords = [kw for kw, _ in top_keywords]

    topics = []
    for template, pair in zip(TOPIC_TEMPLATES,
                              itertools.cycle(itertools.combinations(keywords, 2))):
        kw1, kw2 = pair
        topic = template.format(kw1=kw1.title(), kw2=kw2.title())
        if topic not in topics:
            topics.append(topic)
        if len(topics) >= max_topics:
            break

    return topics, top_keywords


# -------------------- PLOTS --------------------

def plot_upload_frequency(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return

    df_monthly = (
        df.set_index("published_at")
          .resample("ME")        # Pandas 3 compatible
          .size()
          .rename("videos")
          .reset_index()
    )

    st.line_chart(df_monthly, x="published_at", y="videos")


def plot_views_over_time(df: pd.DataFrame):
    if df.empty:
        st.info("No videos to plot.")
        return

    st.line_chart(df, x="published_at", y="views")


# -------------------- STREAMLIT APP --------------------

def main():
    st.set_page_config("YouTube Insight Generator", layout="wide")
    st.title("📊 YouTube Video Analysis & Insight Generator")

    with st.sidebar:
        st.header("Inputs")
        api_key = st.text_input("YouTube Data API Key", type="password")
        video_url = st.text_input("YouTube Video URL")
        max_videos = st.slider("Number of recent videos", 10, 100, 50)
        fetch = st.button("Fetch Insights")

    if fetch:
        if not api_key or not video_url:
            st.error("Please provide API key and video URL.")
            return

        with st.spinner("Fetching data from YouTube..."):
            try:
                client = build_client(api_key)
                channel_id = get_channel_id_from_video(client, video_url)
                playlist_id = get_uploads_playlist_id(client, channel_id)
                video_ids = fetch_playlist_video_ids(client, playlist_id, max_videos)

                if not video_ids:
                    st.error("No videos found for this channel.")
                    return

                details = fetch_video_details(client, video_ids)
                df = build_dataframe(details)

            except Exception as e:
                st.error("Failed to fetch YouTube data.")
                st.exception(e)
                return

        # ----- METRICS -----
        metrics = compute_metrics(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Views", f"{metrics['avg_views']:.0f}")
        c2.metric("Avg Likes", f"{metrics['avg_likes']:.0f}")
        c3.metric("Avg Comments", f"{metrics['avg_comments']:.0f}")
        ratio = metrics["like_comment_ratio"]
        c4.metric("Like/Comment Ratio", f"{ratio:.2f}" if math.isfinite(ratio) else "N/A")

        st.subheader("📅 Upload Frequency")
        plot_upload_frequency(df)

        st.subheader("📈 Views Over Time")
        plot_views_over_time(df)

        st.subheader("💡 Next Video Topic Predictions")
        topics, keywords = build_topic_predictions(df)

        for topic in topics:
            st.markdown(f"- **{topic}**")

        # FIX for LargeUtf8 crash
        kw_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
        kw_df["Keyword"] = kw_df["Keyword"].astype(str)
        kw_df["Score"] = pd.to_numeric(kw_df["Score"], errors="coerce").astype(float)

        st.dataframe(kw_df, use_container_width=True)

        st.subheader("📄 Latest Videos")
        st.dataframe(df[["title", "published_at", "views", "likes", "comments"]])


if __name__ == "__main__":
    main()
